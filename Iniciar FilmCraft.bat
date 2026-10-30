@echo off
rem Abre o FilmCraft com dois cliques. Na primeira vez (ou depois de "Atualizar FilmCraft.bat")
rem compila o programa, o que demora; nas outras abre na hora. Cria um atalho na Area de Trabalho.
chcp 65001 >nul
cd /d "%~dp0"
set "EXE=%~dp0target\release\filmcraft.exe"

if exist "%EXE%" goto :open

where cargo >nul 2>nul
if errorlevel 1 (
  echo O Rust ^(cargo^) nao esta instalado. Instale com:
  echo   winget install --id Rustlang.Rustup -e
  echo   winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  echo Depois feche esta janela e abra este arquivo de novo.
  echo Ou baixe o programa pronto em Releases no GitHub, sem precisar compilar.
  pause
  exit /b 1
)

echo Compilando o FilmCraft pela primeira vez. Isso pode levar de 10 a 40 minutos...
cargo build --release -p filmcraft --features whisper
if errorlevel 1 (
  echo.
  echo A compilacao falhou. Copie a mensagem de erro acima e peca ajuda.
  pause
  exit /b 1
)

:open
set "LNK=%USERPROFILE%\Desktop\FilmCraft.lnk"
if not exist "%LNK%" powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%LNK%'); $s.TargetPath='%EXE%'; $s.WorkingDirectory='%~dp0'; $s.Save()" >nul 2>nul
start "" "%EXE%"
