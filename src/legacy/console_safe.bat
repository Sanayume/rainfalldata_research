@echo off
:: filepath: f:\rainfalldata\console_safe.bat
:: Windows控制台编码修复与安全启动批处理文件

echo =============================================
echo      安全启动降雨预测训练程序
echo =============================================
echo.

:: 设置控制台为UTF-8模式
chcp 65001 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 警告: 无法设置控制台为UTF-8模式
) else (
    echo 已设置控制台为UTF-8模式
)

:: 设置环境变量
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8

echo 环境变量已设置:
echo - PYTHONIOENCODING=utf-8
echo - PYTHONLEGACYWINDOWSSTDIO=utf-8
echo.

echo 可用的启动选项:
echo 1. 运行新训练
echo 2. 继续已有训练
echo 3. 超参数优化
echo 4. 退出
echo.

set /p option="请选择操作 (1-4): "

if "%option%"=="1" (
    echo.
    echo 启动新训练...
    python run_training.py
) else if "%option%"=="2" (
    echo.
    echo 继续已有训练...
    python run_continued_training.py
) else if "%option%"=="3" (
    echo.
    echo 启动超参数优化...
    python optimize_hyperparams.py
) else if "%option%"=="4" (
    echo.
    echo 退出程序...
    goto :end
) else (
    echo.
    echo 无效选项!
    goto :end
)

:end
echo.
echo =============================================
echo 按任意键退出...
pause >nul