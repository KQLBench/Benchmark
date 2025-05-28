# Atomic Red Team Tests Component

This document describes the setup and execution of Atomic Red Team tests within the KQLBench environment. It covers virtual machine (VM) creation, Microsoft Defender for Endpoint (MDE) configuration for logging-only mode, and the installation of Atomic Red Team.

## Directory Structure

```
AtomicRedTeamTests/
├── chosen_tests.csv      # List of specific Atomic Red Team tests to be executed
├── main.py               # Main script to orchestrate test execution
├── README.md             # This file
├── __init__.py
├── generate_questions/   # Scripts and resources for generating questions based on tests
│   ├── create_questions.py
│   ├── evaluate_tests.txt
│   ├── find_questions.py
│   ├── generate_questions.txt
│   └── __init__.py
├── logs/                 # Directory for storing logs from test executions
├── questions_checked/    # Directory for storing checked or validated questions
└── reports/              # Directory for storing reports generated from test results
```

## 1. Virtual Machine Setup (Windows)

This section details the creation of a Windows VM in Azure, which will serve as the environment for running Atomic Red Team tests.

**Azure CLI Command to Create Windows VM:**

```powershell
az vm create `
  --resource-group wipro `
  --name Windows-VM `
  --location swedencentral `
  --image MicrosoftWindowsDesktop:windows-11:win11-24h2-pro:26100.3194.250210 `
  --admin-username wipro `
  --admin-password "<YOUR_SECURE_PASSWORD>" ` # Store passwords securely, e.g., in Azure Key Vault
  --size Standard_D4s_v3 `
  --os-disk-size-gb 256 `
  --public-ip-sku Standard `
  --nsg-rule RDP `
  --security-type TrustedLaunch `
  --enable-secure-boot true `
  --enable-vtpm true `
  --output json
```

**Notes:**
*   `Standard_D4s_v3`: VM size with 4 cores and 16 GB RAM.
*   `location`: Set to `swedencentral` for consistency with other resources.
*   `name`: VM name (max 15 characters).
*   `admin-password`: **Important:** Replace `"<YOUR_SECURE_PASSWORD>"` with a strong password and consider managing it via Azure Key Vault.
*   It is recommended to configure daily backups for the VM in Azure.

## 2. Microsoft Defender for Endpoint (MDE) Configuration

This section guides you through installing MDE and configuring it for logging-only mode. This means disabling preventive security features while ensuring all activities are logged.

### 2.1. Download and Install MDE

Run the following PowerShell commands on the Windows VM:

**Download MDE Installation Package:**
```powershell
Invoke-WebRequest -Uri "https://aka.ms/MDEClientWindows" -OutFile "$env:TEMP\MDEClientWindows.msi"
```

**Install MDE Client (Silent):**
```powershell
Start-Process "$env:TEMP\MDEClientWindows.msi" -ArgumentList "/quiet /norestart" -Wait
```

### 2.2. Onboard Device to MDE

1.  Log in to the [Microsoft Defender Security Center](https://security.microsoft.com).
2.  Navigate to **Settings > Endpoints > Onboarding**.
3.  Select **Windows 10/11** as the operating system and download the **onboarding package** (a `.zip` file).
4.  Extract the downloaded package on the Windows VM.
5.  Run the onboarding script (typically `WindowsDefenderATPLocalOnboardingScript.cmd`) by double-clicking it or executing it from a command prompt.

**Verify Onboarding Status:**
```powershell
Get-MpComputerStatus | Select-Object AMRunningMode, DefenderEnabled, AntivirusSignatureVersion
```

### 2.3. Disable Preventive Security Measures (Logging Only)

Execute the following PowerShell commands on the Windows VM to disable proactive security features:

**Disable Real-time Protection:**
```powershell
Set-MpPreference -DisableRealtimeMonitoring $true
```

**Disable Cloud-Delivered Protection:**
```powershell
Set-MpPreference -MAPSReporting 0
Set-MpPreference -SubmitSamplesConsent 2 # Corresponds to "Never send"
```

**Disable Behavior Monitoring:**
```powershell
Set-MpPreference -DisableBehaviorMonitoring $true
```

**Disable Network Protection:**
```powershell
Set-MpPreference -EnableNetworkProtection 0 # 0 for Disabled, 1 for Enabled, 2 for Audit mode
```

**Disable Exploit Protection (System-wide):**
```powershell
# This command disables all system-wide process mitigation settings.
# Review carefully before applying to production systems.
Set-ProcessMitigation -System -Disable
```

**Disable Attack Surface Reduction (ASR) Rules:**
```powershell
# Example: Disable a specific ASR rule by ID. 
# To disable all, you would need to iterate through all rule IDs and set their action to Disabled.
# Get-MpPreference | Select-Object -ExpandProperty AttackSurfaceReductionRules_Ids | ForEach-Object { Set-MpPreference -AttackSurfaceReductionRules_Ids $_ -AttackSurfaceReductionRules_Actions Disabled }
Set-MpPreference -AttackSurfaceReductionRules_Ids 75668C1F-73B5-4CF0-BD13-90CDB719D8B4 -AttackSurfaceReductionRules_Actions Disabled
```

**Disable Tamper Protection:**
*Tamper Protection prevents unauthorized changes to security settings. Disabling it should be done with caution.* 
1.  Open **Registry Editor** (`regedit`).
2.  Navigate to: `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows Defender\Features`.
3.  Set the `DWORD` value `TamperProtection` to `0` (Disabled). If it doesn't exist, you may need to create it.
4.  A system restart might be required for the change to take full effect.

### 2.4. Validate Logging Configuration

Verify that MDE is in logging mode and preventive features are off:

```powershell
Get-MpComputerStatus | Format-List AMRunningMode, DefenderEnabled, AntivirusSignatureVersion, IsTamperProtected
```
**Expected Output:**
*   `AMRunningMode`: Should ideally be `Passive Mode` if another AV is active, or reflect that real-time monitoring is off.
*   `DefenderEnabled`: Should be `True` (MDE service is running and logging).
*   `IsTamperProtected`: Should be `False`.

## 3. OpenSSH Server Setup (Optional)

For remote access via SSH, follow these steps on the Windows VM:

**Install OpenSSH Server:**
```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
```

**Start and Configure SSHD Service:**
```powershell
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
Get-Service sshd # Verify service is running and set to Automatic
```

**Firewall Rule for SSH:**
```powershell
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH Server (sshd)" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

**Azure Network Security Group (NSG) Rule for SSH:**

Run this Azure CLI command to allow SSH traffic to the VM from your desired source IP range (or `*` for any, with caution):
```powershell
az network nsg rule create --resource-group wipro --nsg-name Windows-VMNSG --name AllowSSH --priority 100 --direction Inbound --access Allow --protocol Tcp --source-port-ranges '*' --destination-port-ranges 22 --source-address-prefixes '*' --destination-address-prefixes '*'
```
*Note: `Windows-VMNSG` is the default NSG name created with a VM named `Windows-VM`. Adjust if your NSG has a different name.*

## 4. Atomic Red Team Installation

Install Atomic Red Team on the Windows VM using PowerShell:

```powershell
IEX (IWR 'https://raw.githubusercontent.com/redcanaryco/invoke-atomicredteam/master/install-atomicredteam.ps1' -UseBasicParsing);
Install-AtomicRedTeam -getAtomics
```
This command downloads and installs the Invoke-AtomicRedTeam module and then downloads the library of atomic tests.

## Conclusion

After completing these steps, you will have a Windows VM configured with Microsoft Defender for Endpoint in logging-only mode and Atomic Red Team installed, ready for executing security tests for the KQLBench benchmark.