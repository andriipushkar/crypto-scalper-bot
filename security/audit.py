"""
Security Audit Module

Automated security checks for the trading bot.
"""
import os
import re
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from loguru import logger


class Severity(Enum):
    """Security issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Represents a security issue."""
    id: str
    title: str
    severity: Severity
    category: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: str = ""
    cwe_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id,
        }


@dataclass
class AuditReport:
    """Security audit report."""
    timestamp: datetime
    issues: List[SecurityIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)

    def add_issue(self, issue: SecurityIssue) -> None:
        self.issues.append(issue)
        severity = issue.severity.value
        self.summary[severity] = self.summary.get(severity, 0) + 1

    @property
    def has_critical(self) -> bool:
        return self.summary.get("critical", 0) > 0

    @property
    def has_high(self) -> bool:
        return self.summary.get("high", 0) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "total_issues": len(self.issues),
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "issues": [i.to_dict() for i in self.issues],
        }


class SecurityAuditor:
    """Performs security audits on the codebase."""

    # Patterns that indicate potential security issues
    SECRET_PATTERNS = [
        (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'private[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded private key"),
        (r'token\s*=\s*["\'][A-Za-z0-9]{20,}["\']', "Hardcoded token"),
        (r'aws[_-]?access[_-]?key[_-]?id\s*=\s*["\'][A-Z0-9]{20}["\']', "AWS Access Key"),
        (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', "Private key in code"),
    ]

    SQL_INJECTION_PATTERNS = [
        (r'execute\([^)]*%[^)]*\)', "Potential SQL injection (string formatting)"),
        (r'execute\([^)]*\.format\([^)]*\)', "Potential SQL injection (format)"),
        (r'execute\([^)]*f"[^"]*{[^}]*}[^"]*"', "Potential SQL injection (f-string)"),
    ]

    COMMAND_INJECTION_PATTERNS = [
        (r'subprocess\.\w+\([^)]*shell\s*=\s*True', "Shell injection risk"),
        (r'os\.system\(', "Potential command injection"),
        (r'os\.popen\(', "Potential command injection"),
        (r'eval\(', "Dangerous eval() usage"),
        (r'exec\(', "Dangerous exec() usage"),
    ]

    UNSAFE_PATTERNS = [
        (r'pickle\.loads?\(', "Unsafe deserialization (pickle)"),
        (r'yaml\.load\([^)]*\)', "Unsafe YAML loading (use safe_load)"),
        (r'verify\s*=\s*False', "SSL verification disabled"),
        (r'CORS\([^)]*origins\s*=\s*\[?\s*["\*\']', "CORS allows all origins"),
    ]

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.report = AuditReport(timestamp=datetime.now())

    def run_full_audit(self) -> AuditReport:
        """Run complete security audit."""
        logger.info("Starting security audit...")

        self._check_secrets()
        self._check_sql_injection()
        self._check_command_injection()
        self._check_unsafe_patterns()
        self._check_dependencies()
        self._check_config_security()
        self._check_file_permissions()
        self._check_api_security()

        self._generate_summary()

        logger.info(f"Audit complete: {len(self.report.issues)} issues found")
        return self.report

    def _scan_files(
        self,
        patterns: List[Tuple[str, str]],
        category: str,
        severity: Severity,
        extensions: List[str] = None,
    ) -> None:
        """Scan files for patterns."""
        extensions = extensions or [".py"]

        for ext in extensions:
            for file_path in self.project_root.rglob(f"*{ext}"):
                if self._should_skip(file_path):
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    lines = content.split('\n')

                    for pattern, description in patterns:
                        for i, line in enumerate(lines, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                # Skip if in comments
                                stripped = line.strip()
                                if stripped.startswith('#') or stripped.startswith('//'):
                                    continue

                                self.report.add_issue(SecurityIssue(
                                    id=f"{category}-{len(self.report.issues)+1}",
                                    title=description,
                                    severity=severity,
                                    category=category,
                                    description=f"Found: {description}",
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=i,
                                    recommendation=f"Review and fix: {line.strip()[:100]}",
                                ))
                except Exception as e:
                    logger.warning(f"Error scanning {file_path}: {e}")

    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.tox'}
        skip_files = {'audit.py', 'test_', 'conftest.py'}

        for part in path.parts:
            if part in skip_dirs:
                return True

        for pattern in skip_files:
            if pattern in path.name:
                return True

        return False

    def _check_secrets(self) -> None:
        """Check for hardcoded secrets."""
        logger.info("Checking for hardcoded secrets...")
        self._scan_files(
            self.SECRET_PATTERNS,
            "secrets",
            Severity.CRITICAL,
            extensions=[".py", ".yaml", ".yml", ".json", ".env.example"],
        )

        # Check for .env files that shouldn't be committed
        for env_file in self.project_root.rglob(".env"):
            if ".git" not in str(env_file):
                self.report.add_issue(SecurityIssue(
                    id=f"secrets-env-{len(self.report.issues)+1}",
                    title=".env file found",
                    severity=Severity.HIGH,
                    category="secrets",
                    description="Environment file with potential secrets found",
                    file_path=str(env_file.relative_to(self.project_root)),
                    recommendation="Ensure .env is in .gitignore",
                ))

    def _check_sql_injection(self) -> None:
        """Check for SQL injection vulnerabilities."""
        logger.info("Checking for SQL injection...")
        self._scan_files(
            self.SQL_INJECTION_PATTERNS,
            "sql-injection",
            Severity.HIGH,
        )

    def _check_command_injection(self) -> None:
        """Check for command injection vulnerabilities."""
        logger.info("Checking for command injection...")
        self._scan_files(
            self.COMMAND_INJECTION_PATTERNS,
            "command-injection",
            Severity.HIGH,
        )

    def _check_unsafe_patterns(self) -> None:
        """Check for unsafe code patterns."""
        logger.info("Checking for unsafe patterns...")
        self._scan_files(
            self.UNSAFE_PATTERNS,
            "unsafe-code",
            Severity.MEDIUM,
        )

    def _check_dependencies(self) -> None:
        """Check for vulnerable dependencies."""
        logger.info("Checking dependencies...")

        # Check if safety is installed
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:10]:  # Limit to first 10
                    self.report.add_issue(SecurityIssue(
                        id=f"deps-outdated-{pkg['name']}",
                        title=f"Outdated dependency: {pkg['name']}",
                        severity=Severity.LOW,
                        category="dependencies",
                        description=f"{pkg['name']} {pkg['version']} -> {pkg['latest_version']}",
                        recommendation=f"pip install --upgrade {pkg['name']}",
                    ))
        except Exception as e:
            logger.warning(f"Could not check outdated packages: {e}")

        # Try running safety check
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities.get("vulnerabilities", []):
                    self.report.add_issue(SecurityIssue(
                        id=f"deps-vuln-{vuln.get('package_name')}-{vuln.get('vulnerability_id')}",
                        title=f"Vulnerable: {vuln.get('package_name')}",
                        severity=Severity.HIGH,
                        category="dependencies",
                        description=vuln.get("advisory", ""),
                        recommendation=f"Upgrade {vuln.get('package_name')} to a safe version",
                        cwe_id=vuln.get("cwe"),
                    ))
        except FileNotFoundError:
            logger.info("Safety not installed, skipping vulnerability scan")
        except Exception as e:
            logger.warning(f"Could not run safety check: {e}")

    def _check_config_security(self) -> None:
        """Check configuration security."""
        logger.info("Checking configuration security...")

        # Check for debug mode in production configs
        for config_file in self.project_root.rglob("*.yaml"):
            if "prod" in str(config_file).lower():
                try:
                    content = config_file.read_text()
                    if re.search(r'debug:\s*true', content, re.IGNORECASE):
                        self.report.add_issue(SecurityIssue(
                            id=f"config-debug-{len(self.report.issues)+1}",
                            title="Debug mode in production config",
                            severity=Severity.HIGH,
                            category="configuration",
                            description="Debug mode should be disabled in production",
                            file_path=str(config_file.relative_to(self.project_root)),
                            recommendation="Set debug: false in production configs",
                        ))
                except Exception:
                    pass

        # Check Kubernetes secrets
        for k8s_file in self.project_root.rglob("*.yaml"):
            if "k8s" in str(k8s_file) or "kubernetes" in str(k8s_file):
                try:
                    content = k8s_file.read_text()
                    if "kind: Secret" in content and "stringData:" in content:
                        self.report.add_issue(SecurityIssue(
                            id=f"config-k8s-secret-{len(self.report.issues)+1}",
                            title="Plain text secrets in K8s manifest",
                            severity=Severity.MEDIUM,
                            category="configuration",
                            description="K8s secrets should use sealed-secrets or external secrets",
                            file_path=str(k8s_file.relative_to(self.project_root)),
                            recommendation="Use SealedSecrets or ExternalSecrets operator",
                        ))
                except Exception:
                    pass

    def _check_file_permissions(self) -> None:
        """Check file permissions."""
        logger.info("Checking file permissions...")

        sensitive_patterns = ["*.pem", "*.key", "*.crt", "*.p12"]

        for pattern in sensitive_patterns:
            for file_path in self.project_root.rglob(pattern):
                try:
                    mode = file_path.stat().st_mode & 0o777
                    if mode & 0o077:  # Group or other has access
                        self.report.add_issue(SecurityIssue(
                            id=f"perms-{len(self.report.issues)+1}",
                            title=f"Insecure file permissions: {file_path.name}",
                            severity=Severity.MEDIUM,
                            category="permissions",
                            description=f"File {file_path.name} has mode {oct(mode)}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation=f"chmod 600 {file_path}",
                        ))
                except Exception:
                    pass

    def _check_api_security(self) -> None:
        """Check API security patterns."""
        logger.info("Checking API security...")

        api_issues = [
            (r'@app\.(get|post|put|delete|patch)\([^)]*\)\s*\n\s*async def \w+\([^)]*\):', "Missing authentication decorator"),
        ]

        # Check for rate limiting
        has_rate_limit = False
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip(py_file):
                continue
            try:
                content = py_file.read_text()
                if "slowapi" in content or "ratelimit" in content.lower():
                    has_rate_limit = True
                    break
            except Exception:
                pass

        if not has_rate_limit:
            self.report.add_issue(SecurityIssue(
                id="api-rate-limit",
                title="No rate limiting detected",
                severity=Severity.MEDIUM,
                category="api-security",
                description="API should have rate limiting to prevent abuse",
                recommendation="Implement rate limiting using slowapi or similar",
            ))

    def _generate_summary(self) -> None:
        """Generate audit summary."""
        checks = {
            "Hardcoded Secrets": "secrets" not in [i.category for i in self.report.issues if i.severity == Severity.CRITICAL],
            "SQL Injection": "sql-injection" not in [i.category for i in self.report.issues],
            "Command Injection": "command-injection" not in [i.category for i in self.report.issues],
            "Unsafe Patterns": sum(1 for i in self.report.issues if i.category == "unsafe-code") < 3,
            "Dependencies": sum(1 for i in self.report.issues if i.category == "dependencies" and i.severity == Severity.HIGH) == 0,
            "Configuration": "configuration" not in [i.category for i in self.report.issues if i.severity == Severity.HIGH],
        }

        for check, passed in checks.items():
            if passed:
                self.report.passed_checks.append(check)
            else:
                self.report.failed_checks.append(check)

    def generate_report_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# Security Audit Report

**Date:** {self.report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Severity | Count |
|----------|-------|
| Critical | {self.report.summary.get('critical', 0)} |
| High     | {self.report.summary.get('high', 0)} |
| Medium   | {self.report.summary.get('medium', 0)} |
| Low      | {self.report.summary.get('low', 0)} |
| Info     | {self.report.summary.get('info', 0)} |

**Total Issues:** {len(self.report.issues)}

## Checks

### Passed
{chr(10).join(f'- {check}' for check in self.report.passed_checks) or '- None'}

### Failed
{chr(10).join(f'- {check}' for check in self.report.failed_checks) or '- None'}

## Issues

"""
        # Group by severity
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            issues = [i for i in self.report.issues if i.severity == severity]
            if issues:
                md += f"\n### {severity.value.upper()}\n\n"
                for issue in issues:
                    md += f"""#### {issue.title}
- **Category:** {issue.category}
- **File:** {issue.file_path or 'N/A'}:{issue.line_number or ''}
- **Description:** {issue.description}
- **Recommendation:** {issue.recommendation}

"""

        return md


def run_security_audit(project_root: str = ".") -> AuditReport:
    """Run security audit and return report."""
    auditor = SecurityAuditor(project_root)
    return auditor.run_full_audit()
