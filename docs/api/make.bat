@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=..\..\venv\Scripts\sphinx-build.exe
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O%
echo.
echo.Build finished. The HTML pages are in %BUILDDIR%/html.
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
if exist api rmdir /s /q api
echo.
echo.Cleanup completed.
goto end

:apidoc
if not exist api mkdir api
..\..\venv\Scripts\sphinx-apidoc.exe -o api/ ../../src/
echo.
echo.API documentation generated in api/ directory.
goto end

:rebuild
call :clean
call :apidoc
call :html
echo.
echo.Full rebuild completed.
goto end

:end
popd