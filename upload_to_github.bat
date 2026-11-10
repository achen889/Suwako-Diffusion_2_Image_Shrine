@echo off
setlocal
cd /d "%~dp0"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This folder is not a Git repository.
    exit /b 1
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo No GitHub remote named "origin" is configured.
    echo Add it with: git remote add origin YOUR_GITHUB_REPOSITORY_URL
    exit /b 1
)

git add -A
if errorlevel 1 (
    echo Could not stage changes.
    exit /b 1
)

echo.
echo Files staged for upload:
git status --short
git diff --cached --quiet
if errorlevel 2 (
    echo Could not inspect staged changes.
    exit /b 1
)
if errorlevel 1 goto commit_changes

echo.
echo No new file changes to commit; pushing any existing local commits.
goto push_changes

:commit_changes
set "COMMIT_MESSAGE=%~1"
if not defined COMMIT_MESSAGE set /p "COMMIT_MESSAGE=Commit message: "
if not defined COMMIT_MESSAGE (
    echo A commit message is required.
    exit /b 1
)

echo.
choice /C YN /M "Commit these files and push to origin"
if errorlevel 2 (
    echo Upload cancelled. Staged changes remain staged.
    exit /b 0
)

git commit -m "%COMMIT_MESSAGE%"
if errorlevel 1 (
    echo Commit failed.
    exit /b 1
)

:push_changes
git push -u origin HEAD
if errorlevel 1 (
    echo Push failed. Review the GitHub remote and authentication, then retry.
    exit /b 1
)

echo Upload complete.
exit /b 0
