@echo off
setlocal
cd /d "%~dp0"

REM ============================================================
REM  绿风润耕 智慧大棚总控 V4.2.1 启动器
REM  自动探测 AI Python 环境；未命中时回退到本地常用路径
REM ============================================================

set "PS_SCRIPT=%~dp0tools\GreenWindConsoleV4.ps1"
set "FOUND_PY="

REM 1) 优先使用环境变量 GREENWIND_PYTHON
if defined GREENWIND_PYTHON (
  if exist "%GREENWIND_PYTHON%" set "FOUND_PY=%GREENWIND_PYTHON%"
)

REM 2) 检查常见 conda / venv 安装位置
if not defined FOUND_PY (
  for %%P in (
    "C:\greenwind_ai_env\Scripts\pythonw.exe"
    "D:\anaconda3\Anaconda_envs\envs\greenwind_ai_env\pythonw.exe"
    "C:\ProgramData\anaconda3\envs\greenwind_ai_env\pythonw.exe"
    "%USERPROFILE%\anaconda3\envs\greenwind_ai_env\pythonw.exe"
    "%~dp0.venv\Scripts\pythonw.exe"
  ) do (
    if exist %%P set "FOUND_PY=%%~P"
  )
)

REM 3) 仍未找到则给出明确指引
if not defined FOUND_PY (
  echo.
  echo [ERROR] 未找到 AI Python 环境。
  echo.
  echo 请任选一种方式解决：
  echo   1) 安装 conda 环境 greenwind_ai_env
  echo   2) 在项目根目录创建 .venv 虚拟环境
  echo   3) 设置环境变量指向你的 pythonw.exe：
  echo      set GREENWIND_PYTHON=D:\path\to\envs\greenwind_ai_env\pythonw.exe
  echo.
  echo 所需依赖：torch, numpy（详见 README）
  echo.
  pause
  exit /b 1
)

echo [OK] AI Python: %FOUND_PY%
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -AiPythonPath "%FOUND_PY%"

if errorlevel 1 (
  echo.
  echo [ERROR] V4 frontend failed to start. Please keep this window and send the red error text.
  pause
)
endlocal