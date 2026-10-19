param(
    [string]$RootPath = (Split-Path -Parent $PSScriptRoot),
    [string]$AiPythonPath = "C:\greenwind_ai_env\Scripts\pythonw.exe",
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
[System.Windows.Forms.Application]::EnableVisualStyles()

$RootPath = [System.IO.Path]::GetFullPath(([string]$RootPath).Trim().TrimEnd([char[]]@('\','/')))
$RunnerPath = Join-Path $RootPath "tools\ai_pipeline\interactive_runtime_v2.py"
$ConfigPath = Join-Path $RootPath "config\ui_demo_config.json"
$StateRoot = Join-Path $RootPath "var\greenhouse_ui"
$StateFile = Join-Path $StateRoot "state.json"
$CommandDir = Join-Path $StateRoot "commands"
$StopFile = Join-Path $StateRoot "stop.request"
$TraceFile = Join-Path $StateRoot "interaction_trace.csv"
New-Item -ItemType Directory -Force -Path $CommandDir | Out-Null

$script:RuntimeProcess = $null
$script:LastStep = -1
$script:LastState = $null
$script:ManualInitialized = $false

$ColorBg = [System.Drawing.Color]::FromArgb(244,247,246)
$ColorPanel = [System.Drawing.Color]::White
$ColorGreen = [System.Drawing.Color]::FromArgb(22,128,91)
$ColorDark = [System.Drawing.Color]::FromArgb(31,42,46)
$ColorMuted = [System.Drawing.Color]::FromArgb(99,113,118)
$ColorBlue = [System.Drawing.Color]::FromArgb(31,111,235)
$ColorOrange = [System.Drawing.Color]::FromArgb(231,126,34)
$ColorRed = [System.Drawing.Color]::FromArgb(198,50,52)
$ColorLine = [System.Drawing.Color]::FromArgb(222,229,226)

function New-Label([string]$Text, [int]$X, [int]$Y, [int]$W, [int]$H, [float]$Size = 10, [bool]$Bold = $false) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Location = New-Object System.Drawing.Point($X,$Y)
    $label.Size = New-Object System.Drawing.Size($W,$H)
    $style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    $label.Font = New-Object System.Drawing.Font("Microsoft YaHei UI",$Size,$style)
    $label.ForeColor = $ColorDark
    $label.BackColor = [System.Drawing.Color]::Transparent
    return $label
}

function New-Button([string]$Text, [int]$X, [int]$Y, [int]$W, [int]$H, [System.Drawing.Color]$Back = $ColorGreen) {
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object System.Drawing.Point($X,$Y)
    $button.Size = New-Object System.Drawing.Size($W,$H)
    $button.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $button.FlatAppearance.BorderSize = 0
    $button.BackColor = $Back
    $button.ForeColor = [System.Drawing.Color]::White
    $button.Font = New-Object System.Drawing.Font("Microsoft YaHei UI",10,[System.Drawing.FontStyle]::Bold)
    $button.Cursor = [System.Windows.Forms.Cursors]::Hand
    return $button
}

function New-Panel([int]$X, [int]$Y, [int]$W, [int]$H) {
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Location = New-Object System.Drawing.Point($X,$Y)
    $panel.Size = New-Object System.Drawing.Size($W,$H)
    $panel.BackColor = $ColorPanel
    $panel.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    return $panel
}

function New-ValueCard([System.Windows.Forms.Control]$Parent, [string]$Title, [string]$Unit, [int]$X, [int]$Y, [int]$W = 180) {
    $p = New-Panel $X $Y $W 86
    $titleLabel = New-Label $Title 12 10 ($W-24) 22 9 $false
    $titleLabel.ForeColor = $ColorMuted
    $valueLabel = New-Label "--" 12 34 ($W-82) 38 19 $true
    $unitLabel = New-Label $Unit ($W-74) 47 66 24 9 $false
    $unitLabel.TextAlign = [System.Drawing.ContentAlignment]::MiddleRight
    $unitLabel.ForeColor = $ColorMuted
    $p.Controls.AddRange(@($titleLabel,$valueLabel,$unitLabel))
    $Parent.Controls.Add($p)
    return $valueLabel
}

function Safe-Text($Value, [string]$Fallback = "--") {
    if ($null -eq $Value) { return $Fallback }
    return [string]$Value
}

function Number-Text($Value, [int]$Digits = 1) {
    if ($null -eq $Value) { return "--" }
    try { return ([double]$Value).ToString("F$Digits") } catch { return "--" }
}

function Write-UiCommand([string]$Type, [hashtable]$Body = @{}) {
    $payload = @{ type = $Type; request_id = [guid]::NewGuid().ToString("N"); created_at = [DateTime]::UtcNow.ToString("o") }
    foreach ($key in $Body.Keys) { $payload[$key] = $Body[$key] }
    $name = "{0}_{1}.json" -f ([DateTime]::UtcNow.ToString("yyyyMMddHHmmssfff")), $payload.request_id
    $final = Join-Path $CommandDir $name
    $temp = "$final.tmp"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temp -Encoding UTF8
    Move-Item -LiteralPath $temp -Destination $final -Force
}

function New-ChartSeries([System.Windows.Forms.DataVisualization.Charting.Chart]$Chart, [string]$Name, [System.Drawing.Color]$Color, [int]$Width = 2, [string]$Dash = "Solid") {
    $s = New-Object System.Windows.Forms.DataVisualization.Charting.Series
    $s.Name = $Name
    $s.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
    $s.BorderWidth = $Width
    $s.Color = $Color
    $s.BorderDashStyle = [System.Enum]::Parse(
        [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle],
        $Dash
    )
    $s.ChartArea = "Main"
    $s.Legend = "Legend"
    [void]$Chart.Series.Add($s)
}

function New-TemperatureChart([int]$X, [int]$Y, [int]$W, [int]$H) {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Location = New-Object System.Drawing.Point($X,$Y)
    $chart.Size = New-Object System.Drawing.Size($W,$H)
    $chart.BackColor = $ColorPanel
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
    $area.Name = "Main"
    $area.BackColor = $ColorPanel
    $area.AxisX.Title = "控制周期"
    $area.AxisY.Title = "温度（℃）"
    $area.AxisX.MajorGrid.LineColor = $ColorLine
    $area.AxisY.MajorGrid.LineColor = $ColorLine
    [void]$chart.ChartAreas.Add($area)
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
    $legend.Name = "Legend"
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
    [void]$chart.Legends.Add($legend)
    New-ChartSeries $chart "实测棚温" $ColorGreen 3 "Solid"
    New-ChartSeries $chart "10分钟预测" $ColorBlue 2 "Dash"
    New-ChartSeries $chart "30分钟预测" $ColorOrange 2 "Dash"
    New-ChartSeries $chart "目标下限" ([System.Drawing.Color]::FromArgb(130,150,150)) 1 "Dot"
    New-ChartSeries $chart "目标上限" ([System.Drawing.Color]::FromArgb(130,150,150)) 1 "Dot"
    return $chart
}

function New-EnergyChart([int]$X, [int]$Y, [int]$W, [int]$H) {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Location = New-Object System.Drawing.Point($X,$Y)
    $chart.Size = New-Object System.Drawing.Size($W,$H)
    $chart.BackColor = $ColorPanel
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
    $area.Name = "Main"
    $area.BackColor = $ColorPanel
    $area.AxisX.Title = "控制周期"
    $area.AxisY.Title = "功率（kW）"
    $area.AxisX.MajorGrid.LineColor = $ColorLine
    $area.AxisY.MajorGrid.LineColor = $ColorLine
    [void]$chart.ChartAreas.Add($area)
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
    $legend.Name = "Legend"
    $legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
    [void]$chart.Legends.Add($legend)
    New-ChartSeries $chart "设备总负荷" $ColorDark 3 "Solid"
    New-ChartSeries $chart "市电取电" $ColorBlue 2 "Solid"
    New-ChartSeries $chart "清洁能源" $ColorGreen 2 "Solid"
    New-ChartSeries $chart "储能功率" $ColorOrange 2 "Dash"
    return $chart
}

function Add-DeviceRows([System.Windows.Forms.DataGridView]$Grid, $Devices) {
    $Grid.Rows.Clear()
    $names = [ordered]@{ heater="加热"; vent="通风口"; curtain="保温帘关闭度"; fan="循环风机" }
    foreach ($key in $names.Keys) {
        $d = $Devices.$key
        [void]$Grid.Rows.Add($names[$key], (Number-Text $d.requested_pct), (Number-Text $d.safe_pct), (Number-Text $d.actual_pct))
    }
}

function New-DeviceGrid([int]$X, [int]$Y, [int]$W, [int]$H) {
    $grid = New-Object System.Windows.Forms.DataGridView
    $grid.Location = New-Object System.Drawing.Point($X,$Y)
    $grid.Size = New-Object System.Drawing.Size($W,$H)
    $grid.BackgroundColor = $ColorPanel
    $grid.BorderStyle = [System.Windows.Forms.BorderStyle]::None
    $grid.AllowUserToAddRows = $false
    $grid.AllowUserToDeleteRows = $false
    $grid.ReadOnly = $true
    $grid.RowHeadersVisible = $false
    $grid.AutoSizeColumnsMode = [System.Windows.Forms.DataGridViewAutoSizeColumnsMode]::Fill
    $grid.SelectionMode = [System.Windows.Forms.DataGridViewSelectionMode]::FullRowSelect
    [void]$grid.Columns.Add("device","设备")
    [void]$grid.Columns.Add("requested","原始请求 %")
    [void]$grid.Columns.Add("safe","安全裁决 %")
    [void]$grid.Columns.Add("actual","实际反馈 %")
    return $grid
}

function New-ControlSlider([System.Windows.Forms.Control]$Parent, [string]$Title, [int]$Y) {
    $Parent.Controls.Add((New-Label $Title 24 $Y 150 28 11 $true))
    $track = New-Object System.Windows.Forms.TrackBar
    $track.Location = New-Object System.Drawing.Point(175,($Y-4))
    $track.Size = New-Object System.Drawing.Size(480,44)
    $track.Minimum = 0; $track.Maximum = 100; $track.TickFrequency = 10
    $value = New-Label "0 %" 670 $Y 90 28 11 $true
    $valueHandler = { $value.Text = "$($track.Value) %" }.GetNewClosure()
    $track.Add_ValueChanged($valueHandler)
    $Parent.Controls.AddRange(@($track,$value))
    return @{ track=$track; label=$value }
}

$Form = New-Object System.Windows.Forms.Form
$Form.Text = "绿风润耕 · 智慧大棚总控 V4.2.1"
$Form.Size = New-Object System.Drawing.Size(1320,880)
$Form.MinimumSize = New-Object System.Drawing.Size(1024,700)
$Form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$Form.BackColor = $ColorBg
$Form.Font = New-Object System.Drawing.Font("Microsoft YaHei UI",9)
$Form.WindowState = [System.Windows.Forms.FormWindowState]::Maximized
$Form.AutoScaleMode = [System.Windows.Forms.AutoScaleMode]::Dpi

$RootLayout = New-Object System.Windows.Forms.TableLayoutPanel
$RootLayout.Dock = [System.Windows.Forms.DockStyle]::Fill
$RootLayout.RowCount = 2
$RootLayout.ColumnCount = 1
[void]$RootLayout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute,72)))
[void]$RootLayout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent,100)))
[void]$RootLayout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent,100)))
$Form.Controls.Add($RootLayout)

$Header = New-Object System.Windows.Forms.Panel
$Header.Dock = [System.Windows.Forms.DockStyle]::Fill
$Header.BackColor = $ColorDark
$HeaderLayout = New-Object System.Windows.Forms.TableLayoutPanel
$HeaderLayout.Dock = [System.Windows.Forms.DockStyle]::Fill
$HeaderLayout.RowCount = 1
$HeaderLayout.ColumnCount = 3
[void]$HeaderLayout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent,100)))
[void]$HeaderLayout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Absolute,220)))
[void]$HeaderLayout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Absolute,165)))
$Title = New-Label "绿风润耕  智慧大棚总控" 0 0 480 36 18 $true
$Title.ForeColor = [System.Drawing.Color]::White
$Title.Dock = [System.Windows.Forms.DockStyle]::Fill
$Title.Padding = New-Object System.Windows.Forms.Padding(20,16,0,0)
$Status = New-Label "运行层未启动" 0 0 220 28 10 $true
$Status.ForeColor = [System.Drawing.Color]::FromArgb(255,196,91)
$Status.Dock = [System.Windows.Forms.DockStyle]::Fill
$Status.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
$StartButton = New-Button "启动虚拟大棚" 0 0 135 38 $ColorGreen
$StartButton.Dock = [System.Windows.Forms.DockStyle]::Fill
$StartButton.Margin = New-Object System.Windows.Forms.Padding(8,14,16,14)
$HeaderLayout.Controls.Add($Title,0,0)
$HeaderLayout.Controls.Add($Status,1,0)
$HeaderLayout.Controls.Add($StartButton,2,0)
$Header.Controls.Add($HeaderLayout)
$RootLayout.Controls.Add($Header,0,0)

$Tabs = New-Object System.Windows.Forms.TabControl
$Tabs.Dock = [System.Windows.Forms.DockStyle]::Fill
$Tabs.Padding = New-Object System.Drawing.Point(18,8)
$Tabs.Font = New-Object System.Drawing.Font("Microsoft YaHei UI",10,[System.Drawing.FontStyle]::Bold)
$RootLayout.Controls.Add($Tabs,0,1)

function New-Tab([string]$Name) {
    $tab = New-Object System.Windows.Forms.TabPage
    $tab.Text = $Name
    $tab.BackColor = $ColorBg
    $tab.AutoScroll = $true
    $tab.AutoScrollMinSize = New-Object System.Drawing.Size(1230,700)
    [void]$Tabs.TabPages.Add($tab)
    return $tab
}

# 1. 大棚实况
$LiveTab = New-Tab "大棚实况"
$LiveTab.Controls.Add((New-Label "虚拟大棚实时状态" 20 18 300 34 16 $true))
$SourceLabel = New-Label "数据源：--" 850 22 400 26 10 $false
$SourceLabel.TextAlign = [System.Drawing.ContentAlignment]::MiddleRight
$LiveTab.Controls.Add($SourceLabel)
$TempCard = New-ValueCard $LiveTab "中心棚温" "℃" 20 60 185
$HumidityCard = New-ValueCard $LiveTab "相对湿度" "%" 215 60 185
$Co2Card = New-ValueCard $LiveTab "CO₂" "ppm" 410 60 185
$SoilCard = New-ValueCard $LiveTab "基质含水率" "%" 605 60 185
$OutdoorCard = New-ValueCard $LiveTab "室外温度" "℃" 800 60 185
$SolarCard = New-ValueCard $LiveTab "太阳辐射" "W/m²" 995 60 205
$LiveChart = New-TemperatureChart 20 165 790 500
$LiveTab.Controls.Add($LiveChart)
$DetailPanel = New-Panel 830 165 370 235
$DetailPanel.Controls.Add((New-Label "环境全量参数" 16 12 220 28 13 $true))
$DetailText = New-Object System.Windows.Forms.TextBox
$DetailText.Location = New-Object System.Drawing.Point(16,48)
$DetailText.Size = New-Object System.Drawing.Size(335,168)
$DetailText.Multiline = $true; $DetailText.ReadOnly = $true; $DetailText.ScrollBars = "Vertical"
$DetailText.Font = New-Object System.Drawing.Font("Consolas",10)
$DetailPanel.Controls.Add($DetailText)
$LiveTab.Controls.Add($DetailPanel)
$LiveGridPanel = New-Panel 830 414 370 251
$LiveGridPanel.Controls.Add((New-Label "棚端设备状态" 16 10 220 28 13 $true))
$LiveDeviceGrid = New-DeviceGrid 12 44 344 190
$LiveGridPanel.Controls.Add($LiveDeviceGrid)
$LiveTab.Controls.Add($LiveGridPanel)

# 2. 控制中心
$ControlTab = New-Tab "控制中心"
$ControlTab.Controls.Add((New-Label "控制权模式" 20 18 240 34 16 $true))
$ManualMode = New-Button "人工控制" 20 58 145 42 $ColorBlue
$AssistMode = New-Button "AI建议 + 人工确认" 175 58 205 42 $ColorGreen
$AutoMode = New-Button "AI自动托管" 390 58 155 42 $ColorOrange
$ControlTab.Controls.AddRange(@($ManualMode,$AssistMode,$AutoMode))
$ModeHelp = New-Label "所有人工命令和 AI 命令都会先经过范围、互斥和数据有效性检查，再进入虚拟大棚。" 570 62 650 34 10 $false
$ModeHelp.ForeColor = $ColorMuted
$ControlTab.Controls.Add($ModeHelp)
$ManualPanel = New-Panel 20 120 790 415
$ManualPanel.Controls.Add((New-Label "设备手动设定" 20 14 240 34 14 $true))
$HeatSlider = New-ControlSlider $ManualPanel "加热功率" 65
$VentSlider = New-ControlSlider $ManualPanel "通风口开度" 130
$CurtainSlider = New-ControlSlider $ManualPanel "保温帘关闭度" 195
$FanSlider = New-ControlSlider $ManualPanel "循环风机" 260
$ApplyManual = New-Button "应用人工设定" 560 340 190 44 $ColorBlue
$ManualPanel.Controls.Add($ApplyManual)
$ControlTab.Controls.Add($ManualPanel)
$PresetPanel = New-Panel 830 120 370 250
$PresetPanel.Controls.Add((New-Label "一键场景" 18 14 200 32 14 $true))
$WarmButton = New-Button "保温升温" 18 60 155 45 $ColorOrange
$CoolButton = New-Button "通风降温" 188 60 155 45 $ColorBlue
$DryButton = New-Button "通风排湿" 18 120 155 45 $ColorGreen
$StopDevicesButton = New-Button "设备归零" 188 120 155 45 ([System.Drawing.Color]::FromArgb(104,117,122))
$PresetPanel.Controls.AddRange(@($WarmButton,$CoolButton,$DryButton,$StopDevicesButton))
$ControlTab.Controls.Add($PresetPanel)
$EmergencyPanel = New-Panel 830 385 370 150
$EmergencyPanel.Controls.Add((New-Label "异常处置" 18 12 200 30 14 $true))
$EmergencyButton = New-Button "紧急停机" 18 58 155 48 $ColorRed
$ClearEmergencyButton = New-Button "解除停机" 188 58 155 48 $ColorGreen
$EmergencyPanel.Controls.AddRange(@($EmergencyButton,$ClearEmergencyButton))
$ControlTab.Controls.Add($EmergencyPanel)
$ControlStatus = New-Label "等待运行层状态……" 20 560 1180 72 11 $false
$ControlStatus.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
$ControlStatus.Padding = New-Object System.Windows.Forms.Padding(12)
$ControlTab.Controls.Add($ControlStatus)

# 3. AI智能托管
$AiTab = New-Tab "AI智能托管"
$AiTab.Controls.Add((New-Label "AI建议与人工复核" 20 18 340 34 16 $true))
$AiPanel = New-Panel 20 62 790 345
$AiTitle = New-Label "等待 AI 形成建议" 22 18 730 42 17 $true
$AiTitle.ForeColor = $ColorGreen
$AiReason = New-Label "启动虚拟大棚后，系统会综合实测值、Transformer 预测值和 SAC 上层策略。" 22 72 730 70 11 $false
$AiReason.AutoSize = $false
$AiPrediction = New-Label "当前 --℃    10分钟 --℃    30分钟 --℃    目标 --℃" 22 155 730 40 12 $true
$AiApprove = New-Button "批准执行" 22 240 150 50 $ColorGreen
$AiModify = New-Button "按左侧人工值修改" 188 240 210 50 $ColorBlue
$AiReject = New-Button "拒绝并用保底策略" 414 240 210 50 $ColorRed
$AiPanel.Controls.AddRange(@($AiTitle,$AiReason,$AiPrediction,$AiApprove,$AiModify,$AiReject))
$AiTab.Controls.Add($AiPanel)
$AiDevicePanel = New-Panel 830 62 370 345
$AiDevicePanel.Controls.Add((New-Label "建议动作与安全裁决" 16 14 300 30 13 $true))
$AiDeviceGrid = New-DeviceGrid 12 52 344 230
$AiDevicePanel.Controls.Add($AiDeviceGrid)
$AiTab.Controls.Add($AiDevicePanel)
$TechPanel = New-Panel 20 427 1180 220
$TechPanel.Controls.Add((New-Label "技术细节（答辩或研发时查看）" 18 12 400 30 13 $true))
$TechText = New-Object System.Windows.Forms.TextBox
$TechText.Location = New-Object System.Drawing.Point(18,48)
$TechText.Size = New-Object System.Drawing.Size(1140,150)
$TechText.Multiline = $true; $TechText.ReadOnly = $true; $TechText.ScrollBars = "Vertical"
$TechText.Font = New-Object System.Drawing.Font("Consolas",10)
$TechPanel.Controls.Add($TechText)
$AiTab.Controls.Add($TechPanel)

# 4. 能源协同
$EnergyTab = New-Tab "能源协同"
$EnergyTab.Controls.Add((New-Label "清洁能源、储能与市电实时平衡" 20 18 500 34 16 $true))
$SocCard = New-ValueCard $EnergyTab "储能SOC" "%" 20 65 220
$RenewableCard = New-ValueCard $EnergyTab "清洁能源出力" "kW" 255 65 220
$GridPowerCard = New-ValueCard $EnergyTab "市电取电" "kW" 490 65 220
$LoadPowerCard = New-ValueCard $EnergyTab "设备总负荷" "kW" 725 65 220
$BatteryPowerCard = New-ValueCard $EnergyTab "储能功率" "kW" 960 65 240
$EnergyChart = New-EnergyChart 20 175 780 460
$EnergyTab.Controls.Add($EnergyChart)
$EnergyPanel = New-Panel 820 175 380 460
$EnergyPanel.Controls.Add((New-Label "市电场景注入" 18 14 260 32 14 $true))
$GridNormalButton = New-Button "市电正常" 18 58 160 44 $ColorGreen
$GridLimitedButton = New-Button "限功率供电" 193 58 160 44 $ColorOrange
$GridOutageButton = New-Button "模拟市电停电" 18 116 160 44 $ColorRed
$GridAutoButton = New-Button "恢复场景时间表" 193 116 160 44 $ColorBlue
$EnergyPanel.Controls.AddRange(@($GridNormalButton,$GridLimitedButton,$GridOutageButton,$GridAutoButton))
$EnergyCommandStatus = New-Label "请选择市电场景；命令会在当前控制周期生效。" 18 166 335 28 9 $false
$EnergyCommandStatus.ForeColor = $ColorMuted
$EnergyPanel.Controls.Add($EnergyCommandStatus)
$EnergyStatusText = New-Object System.Windows.Forms.TextBox
$EnergyStatusText.Location = New-Object System.Drawing.Point(18,198)
$EnergyStatusText.Size = New-Object System.Drawing.Size(335,227)
$EnergyStatusText.Multiline = $true; $EnergyStatusText.ReadOnly = $true; $EnergyStatusText.ScrollBars = "Vertical"
$EnergyStatusText.Font = New-Object System.Drawing.Font("Microsoft YaHei UI",10)
$EnergyPanel.Controls.Add($EnergyStatusText)
$EnergyTab.Controls.Add($EnergyPanel)

# 5. 实验与报告
$ReportTab = New-Tab "实验与报告"
$ReportTab.Controls.Add((New-Label "运行结果与可追溯证据" 20 18 400 34 16 $true))
$StepCard = New-ValueCard $ReportTab "控制周期" "步" 20 65 220
$ManualCountCard = New-ValueCard $ReportTab "人工命令" "次" 255 65 220
$AiCountCard = New-ValueCard $ReportTab "AI执行" "步" 490 65 220
$OverrideCard = New-ValueCard $ReportTab "安全裁决" "步" 725 65 220
$PredStatusCard = New-ValueCard $ReportTab "预测质量" "" 960 65 240
$ReportChart = New-TemperatureChart 20 175 780 460
$ReportTab.Controls.Add($ReportChart)
$ReportPanel = New-Panel 820 175 380 460
$ReportPanel.Controls.Add((New-Label "实验说明" 18 14 260 32 14 $true))
$ReportText = New-Object System.Windows.Forms.TextBox
$ReportText.Location = New-Object System.Drawing.Point(18,52)
$ReportText.Size = New-Object System.Drawing.Size(342,280)
$ReportText.Multiline = $true; $ReportText.ReadOnly = $true; $ReportText.ScrollBars = "Vertical"
$ReportText.Text = "本页面记录虚拟大棚闭环运行结果。`r`n`r`n实测棚温来自虚拟大棚；10/30分钟曲线来自 Transformer；设备动作由人工或 SAC 生成，再经约束控制与安全裁决。`r`n`r`n能源子系统同步记录设备负荷、清洁能源、市电、储能SOC和未供电量。`r`n`r`n仿真结果不能替代真实大棚现场验证。"
$OpenTrace = New-Button "打开运行记录目录" 18 355 342 48 $ColorBlue
$ReportPanel.Controls.AddRange(@($ReportText,$OpenTrace))
$ReportTab.Controls.Add($ReportPanel)

# 6. 系统与投运
$SystemTab = New-Tab "系统与投运"
$SystemTab.Controls.Add((New-Label "系统连接关系与投运边界" 20 18 420 34 16 $true))
$SystemPanel = New-Panel 20 65 1180 255
$SystemSummary = New-Label "虚拟大棚→Transformer/SAC→设备→环境反馈；设备负荷→EMS→新能源/市电/储能→供电反馈" 22 18 1130 45 12 $true
$SystemPaths = New-Object System.Windows.Forms.TextBox
$SystemPaths.Location = New-Object System.Drawing.Point(22,74)
$SystemPaths.Size = New-Object System.Drawing.Size(1130,150)
$SystemPaths.Multiline = $true; $SystemPaths.ReadOnly = $true; $SystemPaths.ScrollBars = "Vertical"
$SystemPaths.Font = New-Object System.Drawing.Font("Consolas",9)
$SystemPaths.Text = "项目根目录：$RootPath`r`n运行配置：$ConfigPath`r`n状态文件：$StateFile`r`n交互记录：$TraceFile`r`nAI Python：$AiPythonPath"
$SystemPanel.Controls.AddRange(@($SystemSummary,$SystemPaths))
$SystemTab.Controls.Add($SystemPanel)
$BoundaryPanel = New-Panel 20 340 1180 250
$BoundaryPanel.Controls.Add((New-Label "从虚拟大棚切换到真实大棚时，只替换数据/设备适配器" 22 18 800 36 14 $true))
$BoundaryText = New-Label "当前环境策略使用 Transformer＋SAC V3，储能由可解释的规则EMS调度。真实部署时，需将环境传感器、设备、电表、BMS、PCS和新能源逆变器按同一契约接入。SAC V3尚未直接输出储能充放电功率；若升级为环境与能源联合SAC，必须加入SOC等状态并重新训练验证。GNN仍属于多棚扩展。" 22 68 1125 130 11 $false
$BoundaryText.AutoSize = $false
$BoundaryPanel.Controls.Add($BoundaryText)
$SystemTab.Controls.Add($BoundaryPanel)

function Get-ManualTargets {
    return @{
        heater = [double]$HeatSlider.track.Value
        vent = [double]$VentSlider.track.Value
        curtain = [double]$CurtainSlider.track.Value
        fan = [double]$FanSlider.track.Value
    }
}

function Send-Preset([string]$Name) { Write-UiCommand "preset" @{ name=$Name } }

$ManualMode.Add_Click({
    Write-UiCommand "set_mode" @{ mode="manual" }
    $ControlStatus.Text = "正在切换到：人工控制……"
})
$AssistMode.Add_Click({
    Write-UiCommand "set_mode" @{ mode="assist" }
    $ControlStatus.Text = "正在切换到：AI建议＋人工确认……"
})
$AutoMode.Add_Click({
    Write-UiCommand "set_mode" @{ mode="auto" }
    $ControlStatus.Text = "正在切换到：AI自动托管……"
})
$ApplyManual.Add_Click({ Write-UiCommand "manual_targets" @{ targets=(Get-ManualTargets) } })
$WarmButton.Add_Click({ Send-Preset "warm" })
$CoolButton.Add_Click({ Send-Preset "cool" })
$DryButton.Add_Click({ Send-Preset "dehumidify" })
$StopDevicesButton.Add_Click({ Send-Preset "stop" })
$EmergencyButton.Add_Click({ Write-UiCommand "emergency_stop" })
$ClearEmergencyButton.Add_Click({ Write-UiCommand "clear_emergency_stop" })
$OpenTrace.Add_Click({ Start-Process explorer.exe $StateRoot })
function Send-GridStatus([string]$Status, [string]$DisplayName) {
    Write-UiCommand "set_grid_status" @{ status=$Status }
    $EnergyCommandStatus.Text = "已发送：$DisplayName，等待运行层确认……"
    $EnergyCommandStatus.ForeColor = $ColorBlue
}
$GridNormalButton.Add_Click({ Send-GridStatus "NORMAL" "市电正常" })
$GridLimitedButton.Add_Click({ Send-GridStatus "LIMITED" "限功率供电" })
$GridOutageButton.Add_Click({ Send-GridStatus "OUTAGE" "市电停电" })
$GridAutoButton.Add_Click({ Send-GridStatus "AUTO" "恢复场景时间表" })

$AiApprove.Add_Click({
    if ($null -ne $script:LastState.ai.pending) {
        Write-UiCommand "ai_decision" @{ decision_id=$script:LastState.ai.pending.decision_id; decision="approve" }
    }
})
$AiReject.Add_Click({
    if ($null -ne $script:LastState.ai.pending) {
        Write-UiCommand "ai_decision" @{ decision_id=$script:LastState.ai.pending.decision_id; decision="reject" }
    }
})
$AiModify.Add_Click({
    if ($null -ne $script:LastState.ai.pending) {
        Write-UiCommand "ai_decision" @{ decision_id=$script:LastState.ai.pending.decision_id; decision="modify"; targets=(Get-ManualTargets) }
    }
})

function Start-Runtime {
    if (-not (Test-Path -LiteralPath $AiPythonPath)) { [System.Windows.Forms.MessageBox]::Show("找不到 AI Python：`r`n$AiPythonPath"); return }
    if (-not (Test-Path -LiteralPath $RunnerPath)) { [System.Windows.Forms.MessageBox]::Show("找不到统一运行层：`r`n$RunnerPath"); return }
    if (-not (Test-Path -LiteralPath $ConfigPath)) { [System.Windows.Forms.MessageBox]::Show("找不到配置文件：`r`n$ConfigPath"); return }
    if (Test-Path -LiteralPath $StopFile) { Remove-Item -LiteralPath $StopFile -Force }
    $args = @($RunnerPath,"--config",$ConfigPath)
    $script:RuntimeProcess = Start-Process -FilePath $AiPythonPath -ArgumentList $args -WorkingDirectory $RootPath -PassThru
    $Status.Text = "运行层启动中"
    $StartButton.Enabled = $false
}
$StartButton.Add_Click({ Start-Runtime })

function Update-Chart([System.Windows.Forms.DataVisualization.Charting.Chart]$Chart, $Trend) {
    foreach ($series in $Chart.Series) { $series.Points.Clear() }
    foreach ($point in $Trend) {
        $x = [double]$point.step
        [void]$Chart.Series["实测棚温"].Points.AddXY($x,[double]$point.temperature_c)
        [void]$Chart.Series["10分钟预测"].Points.AddXY($x,[double]$point.pred_10min_c)
        [void]$Chart.Series["30分钟预测"].Points.AddXY($x,[double]$point.pred_30min_c)
        [void]$Chart.Series["目标下限"].Points.AddXY($x,[double]$point.target_low_c)
        [void]$Chart.Series["目标上限"].Points.AddXY($x,[double]$point.target_high_c)
    }
}

function Update-EnergyChart([System.Windows.Forms.DataVisualization.Charting.Chart]$Chart, $Trend) {
    foreach ($series in $Chart.Series) { $series.Points.Clear() }
    foreach ($point in $Trend) {
        $x = [double]$point.step
        [void]$Chart.Series["设备总负荷"].Points.AddXY($x,[double]$point.load_power_kw)
        [void]$Chart.Series["市电取电"].Points.AddXY($x,[double]$point.grid_power_kw)
        [void]$Chart.Series["清洁能源"].Points.AddXY($x,[double]$point.renewable_power_kw)
        [void]$Chart.Series["储能功率"].Points.AddXY($x,[double]$point.battery_power_kw)
    }
}

function Update-Ui($State) {
    $script:LastState = $State
    $runtime = $State.runtime; $indoor = $State.environment.indoor; $outdoor = $State.environment.outdoor
    $prediction = $State.prediction; $energy = $State.energy
    $modeName = switch ([string]$runtime.mode) {
        "manual" { "人工控制" }
        "assist" { "AI建议＋人工确认" }
        "auto" { "AI自动托管" }
        default { Safe-Text $runtime.mode }
    }
    $sourceName = switch ([string]$runtime.control_source) {
        "manual" { "人工设备指令" }
        "waiting_human" { "等待人工审核AI建议" }
        "ai_auto" { "AI自动执行" }
        "human_approved_ai" { "人工批准AI建议" }
        "human_modified" { "人工修改AI建议" }
        "human_rejected_fixed" { "拒绝AI并启用保底策略" }
        "human_timeout_fixed" { "审核超时并启用保底策略" }
        "grid_event" { "市电场景切换与能源重平衡" }
        "paused" { "系统暂停" }
        default { Safe-Text $runtime.control_source }
    }
    $Status.Text = "{0} · {1}" -f (Safe-Text $runtime.status), $modeName
    $Status.ForeColor = if ($runtime.emergency_stop) { $ColorRed } elseif ($runtime.status -eq "RUNNING") { [System.Drawing.Color]::FromArgb(99,214,156) } else { [System.Drawing.Color]::FromArgb(255,196,91) }
    $SourceLabel.Text = "数据源：{0}  |  {1}  |  {2}" -f $runtime.source,$runtime.greenhouse_name,$runtime.crop
    $TempCard.Text = Number-Text $indoor.center_temperature_c
    $HumidityCard.Text = Number-Text $indoor.relative_humidity_pct
    $Co2Card.Text = Number-Text $indoor.co2_ppm 0
    $SoilCard.Text = Number-Text $indoor.substrate_moisture_pct
    $OutdoorCard.Text = Number-Text $outdoor.temperature_c
    $SolarCard.Text = Number-Text $outdoor.solar_w_m2 0
    $SocCard.Text = Number-Text $energy.battery_soc_pct
    $RenewableCard.Text = Number-Text $energy.renewable_power_kw
    $GridPowerCard.Text = Number-Text $energy.grid_power_kw
    $LoadPowerCard.Text = Number-Text $energy.load_power_kw
    $BatteryPowerCard.Text = Number-Text $energy.battery_power_kw
    $gridName = switch ([string]$energy.grid_status) {
        "NORMAL" { "市电正常" }
        "LIMITED" { "市电限功率" }
        "OUTAGE" { "市电停电" }
        default { Safe-Text $energy.grid_status }
    }
    $inactiveGridColor = [System.Drawing.Color]::FromArgb(112,124,128)
    $GridNormalButton.BackColor = if ($energy.grid_status -eq "NORMAL") { $ColorGreen } else { $inactiveGridColor }
    $GridLimitedButton.BackColor = if ($energy.grid_status -eq "LIMITED") { $ColorOrange } else { $inactiveGridColor }
    $GridOutageButton.BackColor = if ($energy.grid_status -eq "OUTAGE") { $ColorRed } else { $inactiveGridColor }
    $GridAutoButton.BackColor = if ($null -eq $energy.grid_status_override -or [string]::IsNullOrWhiteSpace([string]$energy.grid_status_override)) { $ColorBlue } else { $inactiveGridColor }
    if ($State.last_command.type -eq "set_grid_status") {
        $EnergyCommandStatus.Text = "已生效：$gridName（控制周期 $($runtime.control_step)）"
        $EnergyCommandStatus.ForeColor = $ColorGreen
    }
    $batteryDirection = if ([double]$energy.battery_power_kw -gt 0.01) { "放电" } elseif ([double]$energy.battery_power_kw -lt -0.01) { "充电" } else { "待机" }
    $EnergyStatusText.Text = @"
当前市电      $gridName
EMS模式       $(Safe-Text $energy.ems_mode)
光伏出力      $(Number-Text $energy.pv_power_kw) kW
风电出力      $(Number-Text $energy.wind_power_kw) kW
储能状态      $batteryDirection
供电比例      $([math]::Round([double]$energy.power_supply_ratio*100,1)) %
未供电功率  $(Number-Text $energy.unserved_power_kw) kW
累计市电      $(Number-Text $energy.cumulative_grid_kwh 2) kWh
累计清洁能源  $(Number-Text $energy.cumulative_renewable_used_kwh 2) kWh

正值储能功率=放电，负值=充电。
场景按钮会立即切换市电状态。
"@
    $DetailText.Text = @"
东侧温度      $(Number-Text $indoor.east_temperature_c) ℃
中心温度      $(Number-Text $indoor.center_temperature_c) ℃
西侧温度      $(Number-Text $indoor.west_temperature_c) ℃
冠层温度      $(Number-Text $indoor.canopy_temperature_c) ℃
基质温度      $(Number-Text $indoor.substrate_temperature_c) ℃
基质含水率    $(Number-Text $indoor.substrate_moisture_pct) %
棚内湿度      $(Number-Text $indoor.relative_humidity_pct) %
CO₂           $(Number-Text $indoor.co2_ppm 0) ppm
室外湿度      $(Number-Text $outdoor.relative_humidity_pct) %
室外风速      $(Number-Text $outdoor.wind_m_s) m/s
天气场景      $(Safe-Text $State.environment.weather_scene)
"@
    Add-DeviceRows $LiveDeviceGrid $State.devices
    $inactiveModeColor = [System.Drawing.Color]::FromArgb(112,124,128)
    $ManualMode.BackColor = if ($runtime.mode -eq "manual") { $ColorBlue } else { $inactiveModeColor }
    $AssistMode.BackColor = if ($runtime.mode -eq "assist") { $ColorGreen } else { $inactiveModeColor }
    $AutoMode.BackColor = if ($runtime.mode -eq "auto") { $ColorOrange } else { $inactiveModeColor }
    $ManualMode.Text = if ($runtime.mode -eq "manual") { "✓ 人工控制" } else { "人工控制" }
    $AssistMode.Text = if ($runtime.mode -eq "assist") { "✓ AI建议 + 人工确认" } else { "AI建议 + 人工确认" }
    $AutoMode.Text = if ($runtime.mode -eq "auto") { "✓ AI自动托管" } else { "AI自动托管" }
    $safetyText = (@($State.safety.current_overrides) -join '；')
    if ([string]::IsNullOrWhiteSpace($safetyText)) { $safetyText = "本周期未触发额外安全修正" }
    $ControlStatus.Text = "当前模式：$modeName　控制来源：$sourceName　仿真时刻：$([math]::Round([double]$runtime.sim_hour,2)) h`r`n安全裁决：$safetyText"

    if (-not $script:ManualInitialized) {
        $HeatSlider.track.Value = [int][math]::Round([double]$State.manual_targets_pct.heater)
        $VentSlider.track.Value = [int][math]::Round([double]$State.manual_targets_pct.vent)
        $CurtainSlider.track.Value = [int][math]::Round([double]$State.manual_targets_pct.curtain)
        $FanSlider.track.Value = [int][math]::Round([double]$State.manual_targets_pct.fan)
        $script:ManualInitialized = $true
    }

    $pending = $State.ai.pending
    if ($null -ne $pending -and $null -ne $pending.decision_id) {
        $AiTitle.Text = Safe-Text $pending.title "AI已形成建议"
        $AiReason.Text = Safe-Text $pending.reason
        $AiApprove.Enabled = $true; $AiModify.Enabled = $true; $AiReject.Enabled = $true
        $requested = $pending.requested_devices_pct; $safe = $pending.safe_devices_pct
        $previewDevices = [pscustomobject]@{}
        foreach ($name in @("heater","vent","curtain","fan")) {
            $previewDevices | Add-Member NoteProperty $name ([pscustomobject]@{ requested_pct=$requested.$name; safe_pct=$safe.$name; actual_pct=$State.devices.$name.actual_pct })
        }
        Add-DeviceRows $AiDeviceGrid $previewDevices
        $TechText.Text = "SAC策略：`r`n$($pending.strategy | ConvertTo-Json -Depth 6)`r`n`r`n规划约束：$((@($pending.planner_constraints) -join '；'))`r`n安全裁决：$((@($pending.safety_overrides) -join '；'))"
    } else {
        $AiTitle.Text = if ($runtime.mode -eq "auto") { "AI自动托管正在运行" } elseif ($runtime.mode -eq "manual") { "当前由人工控制" } else { "等待下一条 AI 建议" }
        $AiReason.Text = "Transformer 负责预测环境变化，SAC 提供上层控制策略，最终动作仍需经过约束控制和安全裁决。"
        $AiApprove.Enabled = $false; $AiModify.Enabled = $false; $AiReject.Enabled = $false
        Add-DeviceRows $AiDeviceGrid $State.devices
        $TechText.Text = "当前技术策略：`r`n$($State.ai.technical_strategy | ConvertTo-Json -Depth 6)`r`n`r`n规划约束：$((@($State.ai.planner_constraints) -join '；'))"
    }
    $band = @($prediction.target_band_c)
    $targetText = if ($band.Count -ge 2) { "{0}–{1}" -f (Number-Text $band[0]),(Number-Text $band[1]) } else { "--" }
    $AiPrediction.Text = "当前 $(Number-Text $indoor.center_temperature_c)℃    10分钟 $(Number-Text $prediction.temperature_10min_c)℃    30分钟 $(Number-Text $prediction.temperature_30min_c)℃    目标 $targetText℃"

    $StepCard.Text = Safe-Text $runtime.control_step "0"
    $ManualCountCard.Text = Safe-Text $State.metrics.manual_command_count "0"
    $AiCountCard.Text = Safe-Text $State.metrics.ai_executed_steps "0"
    $OverrideCard.Text = Safe-Text $State.safety.total_override_steps "0"
    $PredStatusCard.Text = Safe-Text $prediction.quality

    $step = [int]$runtime.control_step
    if ($step -ne $script:LastStep) {
        Update-Chart $LiveChart $State.trend
        Update-Chart $ReportChart $State.trend
        Update-EnergyChart $EnergyChart $State.trend
        $script:LastStep = $step
    }
}

$Timer = New-Object System.Windows.Forms.Timer
$Timer.Interval = 900
$Timer.Add_Tick({
    try {
        if (Test-Path -LiteralPath $StateFile) {
            $text = Get-Content -LiteralPath $StateFile -Raw -Encoding UTF8
            if (-not [string]::IsNullOrWhiteSpace($text)) { Update-Ui ($text | ConvertFrom-Json) }
        }
        if ($null -ne $script:RuntimeProcess -and $script:RuntimeProcess.HasExited) {
            $StartButton.Enabled = $true
            if ($Status.Text -like "RUNNING*") { $Status.Text = "运行层已退出" }
        }
    } catch {
        $Status.Text = "状态刷新等待中"
    }
})
$Timer.Start()

$Form.Add_FormClosing({
    try {
        "stop" | Set-Content -LiteralPath $StopFile -Encoding UTF8
        if ($null -ne $script:RuntimeProcess -and -not $script:RuntimeProcess.HasExited) {
            if (-not $script:RuntimeProcess.WaitForExit(4000)) { $script:RuntimeProcess.Kill() }
        }
    } catch {}
})

if ($SmokeTest) {
    $SmokeTimer = New-Object System.Windows.Forms.Timer
    $SmokeTimer.Interval = 1800
    $SmokeTimer.Add_Tick({ $SmokeTimer.Stop(); $Form.Close() })
    $SmokeTimer.Start()
}

[void]$Form.ShowDialog()
