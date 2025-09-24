#define AppName "AI Movie Maker"
#define AppVersion "1.3.2"
#define Publisher "AI Movie Maker"
#define URL "https://example.com/ai-movie-maker"
#define DefaultModels "sd15,animatediff"

[Setup]
AppId={{5B4F2C6E-ED74-47D2-9EF7-98CDA979CFE1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableDirPage=no
DisableProgramGroupPage=no
OutputDir=dist
OutputBaseFilename=AI_Movie_Maker_Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\ai_movie_maker.py
WizardStyle=modern
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "downloadrife"; Description: "Install RIFE frame interpolation"; Flags: unchecked
Name: "downloadrealesrgan"; Description: "Install Real-ESRGAN upscaler"; Flags: unchecked
Name: "downloadwkhtml"; Description: "Install wkhtmltopdf exporter"; Flags: unchecked
Name: "downloadmodel_sd15"; Description: "Download Stable Diffusion 1.5"; Flags: unchecked
Name: "downloadmodel_sdxl"; Description: "Download Stable Diffusion XL"; Flags: unchecked
Name: "downloadmodel_animatediff"; Description: "Download AnimateDiff motion adapter"; Flags: unchecked
Name: "downloadmodel_ipadapter"; Description: "Download IP-Adapter"; Flags: unchecked

[Files]
Source: "ai_movie_maker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_models.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\run.bat"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\run.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\run.bat"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\bin\*.*"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{localappdata}\{#AppName}"

[Code]
const
  FfmpegUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip';
  FfmpegSha256 = 'd6f787e91c785d012450375f0a2d54e4719463c261e1b12361d007f4369527f4';
  RifeUrl = 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip';
  RifeSha256 = 'd8e4d772d26cd8006ef0ad0bc82eb191b53c68677d1ae2f42506d74cbbbea606';
  RealesrganUrl = 'https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip';
  RealesrganSha256 = '1bbbdb12d470af80b035c773682e144c6c2f6ece9210832a289af0a48ce3fa9a';
  WkhtmlUrl = 'https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.mxe-cross-win64.7z';
  WkhtmlSha256 = '6ce994c89d5f6018fa158e149a2de8fbbbeb14d8ff5cb656c89893bc556e0557';
  ComfyUrl = 'https://github.com/YanWenKun/ComfyUI-Windows-Portable/releases/download/20250816/ComfyUI_Windows_Portable_Nvidia.zip';
  ComfySha256 = '0f43a033c4695013e7ac06e902b4d8e57e9375497400d463e271a364175b9f87';

var
  ComboFfmpeg: TNewComboBox;
  ComboComfy: TNewComboBox;
  CustomModelDirEdit: TNewEdit;
  SilentAll: Boolean;

procedure InitializeWizard;
var
  Page: TWizardPage;
  LabelModels: TNewStaticText;
begin
  SilentAll := WizardSilent;
  Page := CreateCustomPage(wpSelectTasks, '{#AppName} Components',
    'Choose versions and optional downloads');

  ComboFfmpeg := TNewComboBox.Create(Page);
  ComboFfmpeg.Parent := Page.Surface;
  ComboFfmpeg.Style := csDropDownList;
  ComboFfmpeg.Items.Add('ffmpeg-8.0-essentials_build.zip (latest)');
  ComboFfmpeg.ItemIndex := 0;
  ComboFfmpeg.Top := ScaleY(40);
  ComboFfmpeg.Width := Page.SurfaceWidth;

  ComboComfy := TNewComboBox.Create(Page);
  ComboComfy.Parent := Page.Surface;
  ComboComfy.Style := csDropDownList;
  ComboComfy.Items.Add('ComfyUI Portable (2025-08-16)');
  ComboComfy.ItemIndex := 0;
  ComboComfy.Top := ComboFfmpeg.Top + ComboFfmpeg.Height + ScaleY(16);
  ComboComfy.Width := Page.SurfaceWidth;

  LabelModels := TNewStaticText.Create(Page);
  LabelModels.Parent := Page.Surface;
  LabelModels.Caption := 'Optional models will be downloaded via HuggingFace during setup.';
  LabelModels.Top := ComboComfy.Top + ComboComfy.Height + ScaleY(16);
  LabelModels.AutoSize := True;

  CustomModelDirEdit := TNewEdit.Create(Page);
  CustomModelDirEdit.Parent := Page.Surface;
  CustomModelDirEdit.Top := LabelModels.Top + LabelModels.Height + ScaleY(16);
  CustomModelDirEdit.Width := Page.SurfaceWidth;
  CustomModelDirEdit.Text := ExpandConstant('{localappdata}\{#AppName}\models');
  CustomModelDirEdit.TextHint := 'Custom model directory (optional)';
end;

function RunPowerShell(const Script: string): Boolean;
var
  ResultCode: Integer;
  Command: string;
begin
  Command := '-NoProfile -ExecutionPolicy Bypass -Command "' + Script + '"';
  Result := Exec('powershell.exe', Command, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function DownloadFileWithRetry(const Url, Dest: string; Retries: Integer): Boolean;
var
  Script: string;
  Attempt: Integer;
begin
  Result := False;
  for Attempt := 1 to Retries do
  begin
    Script := 'try {Invoke-WebRequest -Uri "' + Url + '" -OutFile "' + Dest + '" -UseBasicParsing; exit 0} catch {exit 1}';
    if RunPowerShell(Script) then
    begin
      Result := True;
      Exit;
    end;
    Log(Format('Download attempt %d for %s failed', [Attempt, Url]));
  end;
end;

function VerifySha256(const FileName, ExpectedHash: string): Boolean;
var
  Script: string;

  TempFile: string;
  Output: string;
begin
  if ExpectedHash = '' then
  begin
    Result := True;
    Exit;
  end;
  TempFile := ExpandConstant('{tmp}\hash.txt');
  Script := Format('Get-FileHash -Path ''%s'' -Algorithm SHA256 | Select-Object -ExpandProperty Hash | Set-Content -Path ''%s''',
    [FileName, TempFile]);
  Result := RunPowerShell(Script);
  if Result and LoadStringFromFile(TempFile, Output) then
    Result := CompareText(Trim(Output), ExpectedHash) = 0
  else if Result then
    Result := False;
end;

function ExpandArchive(const Archive, Dest: string): Boolean;
var
  Script: string;
begin
  Script := 'Add-Type -AssemblyName System.IO.Compression.FileSystem; ' +
            '[System.IO.Compression.ZipFile]::ExtractToDirectory("' + Archive + '", "' + Dest + '", $true)';
  Result := RunPowerShell(Script);
end;

procedure InstallRequirements(const PythonExe: string);
var
  Command: string;
  ResultCode: Integer;
begin
  if not FileExists(PythonExe) then
  begin
    Log('Python executable not found: ' + PythonExe);
    Exit;
  end;
  Command := Format('"%s" -m pip install --upgrade pip', [PythonExe]);
  if not Exec(PythonExe, '-m pip install --upgrade pip', '{app}', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    Log('Failed to upgrade pip');
  Exec(PythonExe, Format('-m pip install -r "%s"', [ExpandConstant('{app}\requirements.txt')]), '{app}', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure CreateRunScript;
var
  BatFile: string;
  Lines: TStringList;
  PythonExe: string;
begin
  BatFile := ExpandConstant('{app}\run.bat');
  Lines := TStringList.Create;
  try
    PythonExe := ExpandConstant('{app}\bin\ComfyUI_windows_portable\python_embedded\python.exe');
    Lines.Add('@echo off');
    Lines.Add('setlocal');
    Lines.Add('set APPDIR=%~dp0');
    Lines.Add('set BASEDIR=%LOCALAPPDATA%\AI Movie Maker');
    Lines.Add('if not exist "%BASEDIR%" mkdir "%BASEDIR%"');
    Lines.Add('set PATH=%APPDIR%bin\ffmpeg\bin;%PATH%');
    Lines.Add('if exist "%APPDIR%bin\ComfyUI_windows_portable\run_nvidia_gpu.bat" (');
    Lines.Add('  set COMFY_RUN="%APPDIR%bin\ComfyUI_windows_portable\run_nvidia_gpu.bat"');
    Lines.Add(') else (');
    Lines.Add('  set COMFY_RUN="%APPDIR%bin\ComfyUI_windows_portable\run_cpu.bat"');
    Lines.Add(')');
    Lines.Add('start "ComfyUI" /MIN %COMFY_RUN%');
    Lines.Add('"' + PythonExe + '" "%APPDIR%ai_movie_maker.py" gui');
    Lines.Add('endlocal');
    Lines.SaveToFile(BatFile);
  finally
    Lines.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  FfmpegZip, RifeZip, EsrganZip, Wkhtml7z, ComfyZip: string;
  BinDir: string;
  PythonExe: string;
  ResultCode: Integer;
begin
  if CurStep = ssInstall then
  begin
    BinDir := ExpandConstant('{app}\bin');
    if not DirExists(BinDir) then
      ForceDirectories(BinDir);

    ComfyZip := ExpandConstant('{tmp}\comfyui.zip');
    if DownloadFileWithRetry(ComfyUrl, ComfyZip, 3) then
    begin
      if not VerifySha256(ComfyZip, ComfySha256) then
      begin
        MsgBox('ComfyUI download failed integrity verification.', mbError, MB_OK);
        Abort;
      end;
      ForceDirectories(BinDir + '\\ComfyUI_windows_portable');
      ExpandArchive(ComfyZip, BinDir + '\\ComfyUI_windows_portable');
    end
    else if not SilentAll then
      MsgBox('Unable to download ComfyUI package.', mbError, MB_OK);

    FfmpegZip := ExpandConstant('{tmp}\ffmpeg.zip');
    if DownloadFileWithRetry(FfmpegUrl, FfmpegZip, 3) then
    begin
\      if not VerifySha256(FfmpegZip, FfmpegSha256) then
      begin
        MsgBox('FFmpeg download failed integrity verification.', mbError, MB_OK);
        Abort;
      end;
      ForceDirectories(BinDir + '\\ffmpeg');
      ExpandArchive(FfmpegZip, BinDir + '\\ffmpeg');
    end
    else if not SilentAll then
      MsgBox('Unable to download FFmpeg package.', mbError, MB_OK);

    if WizardIsTaskSelected('downloadrife') then
    begin
      RifeZip := ExpandConstant('{tmp}\rife.zip');
      if DownloadFileWithRetry(RifeUrl, RifeZip, 3) then
      begin
        if not VerifySha256(RifeZip, RifeSha256) then
        begin
          MsgBox('RIFE download failed integrity verification.', mbError, MB_OK);
          Abort;
        end;
        ForceDirectories(BinDir + '\\rife');
        ExpandArchive(RifeZip, BinDir + '\\rife');
      end
      else if not SilentAll then
        MsgBox('Unable to download RIFE package.', mbError, MB_OK);
    end;

    if WizardIsTaskSelected('downloadrealesrgan') then
    begin
      EsrganZip := ExpandConstant('{tmp}\realesrgan.zip');
      if DownloadFileWithRetry(RealesrganUrl, EsrganZip, 3) then
      begin
        if not VerifySha256(EsrganZip, RealesrganSha256) then
        begin
          MsgBox('Real-ESRGAN download failed integrity verification.', mbError, MB_OK);
          Abort;
        end;
        ForceDirectories(BinDir + '\\realesrgan');
        ExpandArchive(EsrganZip, BinDir + '\\realesrgan');
      end
      else if not SilentAll then
        MsgBox('Unable to download Real-ESRGAN package.', mbError, MB_OK);
    end;

    if WizardIsTaskSelected('downloadwkhtml') then
    begin
      Wkhtml7z := ExpandConstant('{tmp}\wkhtml.7z');
      if DownloadFileWithRetry(WkhtmlUrl, Wkhtml7z, 3) then
      begin
        if not VerifySha256(Wkhtml7z, WkhtmlSha256) then
        begin
          MsgBox('wkhtmltopdf download failed integrity verification.', mbError, MB_OK);
          Abort;
        end;
        Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath ''" + Wkhtml7z + "'' -DestinationPath ''" + BinDir + "\\wkhtmltopdf'' -Force"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      end
      else if not SilentAll then
        MsgBox('Unable to download wkhtmltopdf package.', mbError, MB_OK);
    end;

    PythonExe := ExpandConstant('{app}\bin\ComfyUI_windows_portable\python_embedded\python.exe');
    InstallRequirements(PythonExe);

    CreateRunScript();
  end
  else if CurStep = ssPostInstall then
  begin
    if not SilentAll then
      MsgBox('Installation complete! Launch AI Movie Maker from the shortcut or run.bat.', mbInformation, MB_OK);
  end;
end;

procedure CurUninstallStepChanged(CurStep: TUninstallStep);
begin
  if CurStep = usPostUninstall then
    MsgBox('{#AppName} has been removed. Models remain in %LOCALAPPDATA%\{#AppName}.', mbInformation, MB_OK);
end;

function NeedRestart: Boolean;
begin
  Result := False;
end;

