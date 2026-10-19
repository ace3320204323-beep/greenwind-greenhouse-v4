@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   绿风润耕 V4 整合版 发布自检
echo ============================================================
echo.

set "AIPY=C:\greenwind_ai_env\Scripts\python.exe"
if not exist "%AIPY%" (
  echo [提示] 未找到 C:\greenwind_ai_env\Scripts\python.exe，改用系统 python
  set "AIPY=python"
)
echo 使用 Python: %AIPY%
echo.

echo [1/7] 目录结构与关键文件检查
set "ERR=0"
for %%F in (
  "tools\GreenWindConsoleV4.ps1"
  "tools\ai_pipeline\interactive_runtime_v2.py"
  "tools\ai_pipeline\integrated_runtime_v1.py"
  "tools\ai_pipeline\greenhouse_sac_env.py"
  "tools\ai_pipeline\greenhouse_sac_env_v3.py"
  "tools\ai_pipeline\transformer_inference.py"
  "tools\ai_pipeline\train_transformer_v2.py"
  "tools\ai_pipeline\strategy_to_control_v3.py"
  "tools\ai_pipeline\generate_data_v2.py"
  "tools\ai_pipeline\outputs_v2_1_residual\transformer_v2.pt"
  "tools\ai_pipeline\outputs_v2_1_residual\transformer_v2_meta.json"
  "tools\ai_pipeline\sac_runs\formal_v3\sac_v3_final.zip"
  "tools\virtual_greenhouse\runner.py"
  "tools\virtual_greenhouse\config.py"
  "tools\virtual_greenhouse\energy_system.py"
  "tools\virtual_greenhouse\test_energy_system.py"
  "tools\virtual_greenhouse\energy_storage_experiment.py"
  "tools\virtual_greenhouse\scenarios\shandong_winter_strawberry.json"
  "config\ui_demo_config.json"
) do (
  if exist "%%~F" (echo   OK   %%~F) else (echo   MISS %%~F & set "ERR=1")
)
echo.

echo [2/7] Python 语法编译检查
"%AIPY%" -m compileall -q tools >nul 2>&1
if errorlevel 1 (echo   语法检查失败 & set "ERR=1") else (echo   语法检查通过)
for /d /r %%D in (__pycache__) do @if exist "%%D" rd /s /q "%%D"
echo.

echo [3/7] V4 契约测试（页面/曲线/命令/模型路径）
pushd tools\ai_pipeline
"%AIPY%" test_interactive_v4_contract.py
if errorlevel 1 (set "ERR=1")
echo.
echo [4/7] 统一运行时测试
"%AIPY%" test_integrated_runtime.py
if errorlevel 1 (set "ERR=1")
popd
for /d /r %%D in (__pycache__) do @if exist "%%D" rd /s /q "%%D"
echo.

echo [5/7] 热惯性回归测试
"%AIPY%" tools\virtual_greenhouse\test_thermal_inertia.py
if errorlevel 1 (set "ERR=1")
echo.

echo [6/7] 能源子系统回归测试
"%AIPY%" tools\virtual_greenhouse\test_energy_system.py
if errorlevel 1 (set "ERR=1")
echo.

echo [7/7] 运行层配置与模型路径自检
"%AIPY%" tools\ai_pipeline\interactive_runtime_v2.py --config config\ui_demo_config.json --validate-only
if errorlevel 1 (set "ERR=1")
for /d /r %%D in (__pycache__) do @if exist "%%D" rd /s /q "%%D"
echo.

if "%ERR%"=="1" (
  echo ============================================================
  echo   自检未通过，请保留本窗口并将红色错误信息发送给我们
  echo ============================================================
) else (
  echo ============================================================
  echo   自检全部通过，可以双击 启动绿风润耕V4.bat 运行演示
  echo ============================================================
)
echo.
pause
endlocal
