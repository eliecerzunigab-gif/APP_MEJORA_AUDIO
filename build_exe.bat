@echo off
echo ============================================
echo   Audio Enhancer - Build Installer
echo ============================================
echo.
echo Instalando dependencias necesarias...
pip install pyinstaller --quiet
echo.
echo Generando Audio_Enhancer.exe (esto tomara 2-3 minutos)...
echo.
pyinstaller build.spec --clean --noconfirm
echo.
if exist "dist\Audio_Enhancer.exe" (
    echo ============================================
    echo   EXE generado exitosamente!
    echo   Archivo: dist\Audio_Enhancer.exe
    echo.
    echo   PARA USAR LA APP:
    echo   1. Copia "dist\Audio_Enhancer.exe" a una carpeta
    echo   2. Crea la carpeta "MP3_01" junto al .exe
    echo   3. Pon tus archivos MP3 dentro de MP3_01
    echo   4. Ejecuta Audio_Enhancer.exe
    echo ============================================
) else (
    echo [ERROR] No se pudo generar el .exe
    echo Revisa los mensajes de error arriba.
)
pause