@echo off
rem Baixa as ultimas mudancas do seu fork no GitHub, compila e abre o FilmCraft.
chcp 65001 >nul
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo O Git nao esta instalado. Instale com:  winget install --id Git.Git -e
  pause
  exit /b 1
)
echo Baixando as atualizacoes...
git pull
if errorlevel 1 (
  echo Nao foi possivel atualizar. Copie a mensagem acima e peca ajuda.
  pause
  exit /b 1
)
echo Compilando as mudancas...
cargo build --release -p filmcraft --features whisper
if errorlevel 1 (
  echo A compilacao falhou. Copie a mensagem de erro acima e peca ajuda.
  pause
  exit /b 1
)
start "" "%~dp0target\release\filmcraft.exe"
