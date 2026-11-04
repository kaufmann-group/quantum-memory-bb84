
#!/bin/bash

# Sync Overleaf files to GitHub under docs/
# Compile standalone LaTeX documents and store PDFs in docs/pdfs/

set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
TMPDIR=$(mktemp -d)
BRANCH_NAME="tmp-overleaf-sync-$(date +%s)-$$"

cleanup() {
    git -C "$REPO" worktree remove --force "$TMPDIR" 2>/dev/null || true
    rm -rf "$TMPDIR"
    git -C "$REPO" branch -D "$BRANCH_NAME" 2>/dev/null || true
}
trap cleanup EXIT

# Fetch latest versions
git -C "$REPO" fetch origin main
git -C "$REPO" fetch overleaf master

# Create temporary worktree without changing current branch
git -C "$REPO" worktree add -b "$BRANCH_NAME" "$TMPDIR" origin/main

# Replace docs with Overleaf files
rm -rf "$TMPDIR/docs"
mkdir -p "$TMPDIR/docs/pdfs"

git -C "$REPO" archive refs/remotes/overleaf/master |
    tar -x -C "$TMPDIR/docs"

# Compile LaTeX documents
cd "$TMPDIR/docs"

for file in "Documentation"/*.tex; do
    [ -e "$file" ] || continue

    if grep -Fxq '% git ignore this file' "$file"; then
        echo "Ignoring $file"
        rm -f "$file"
        continue
    fi

    if grep -q '\\documentclass' "$file"; then
        echo "Compiling $file..."

        base_name=$(basename "$file")
        name="${base_name%.tex}"

        # Compile from the document's directory
        (
            cd "$(dirname "$file")"

            pdflatex -shell-escape \
                -interaction=nonstopmode \
                -halt-on-error \
                "$base_name"

            pdflatex -shell-escape \
                -interaction=nonstopmode \
                -halt-on-error \
                "$base_name"

            mv "$name.pdf" ../pdfs/

            rm -rf "_minted-$name"
            rm -f "$name".{aux,log,toc,out,snm,nav,vrb,fls,fdb_latexmk,synctex.gz}
        )
    else
        echo "Skipping non-document file: $file"
    fi
done

# Commit and push
cd "$TMPDIR"

git add -A docs

if git diff --cached --quiet; then
    echo "No changes to commit."
else
    git commit -m "sync docs and PDFs from Overleaf"
    git push origin HEAD:main
fi

echo "Overleaf synchronization complete."
