"""
Тести для прикладів конфігурацій та Grafana дашбордів.
"""

import json
import pytest
from pathlib import Path

import yaml


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def project_root() -> Path:
    """Повертає кореневу директорію проекту."""
    return Path(__file__).parent.parent


@pytest.fixture
def examples_dir(project_root) -> Path:
    """Повертає директорію з прикладами конфігурацій."""
    return project_root / "examples"


@pytest.fixture
def grafana_dir(project_root) -> Path:
    """Повертає директорію з Grafana дашбордами."""
    return project_root / "grafana"


# =============================================================================
# Example Configs Tests
# =============================================================================

class TestExampleConfigs:
    """Тести для прикладів конфігурацій."""

    def test_examples_directory_exists(self, examples_dir):
        """Перевірка існування директорії examples."""
        assert examples_dir.exists(), "Директорія examples не знайдена"

    def test_all_example_configs_exist(self, examples_dir):
        """Перевірка наявності всіх прикладів конфігурацій."""
        expected_configs = [
            "config-conservative.yaml",
            "config-aggressive.yaml",
            "config-multi-exchange.yaml",
            "config-ml-focused.yaml",
        ]

        for config_name in expected_configs:
            config_path = examples_dir / config_name
            assert config_path.exists(), f"Конфігурація {config_name} не знайдена"

    def test_example_configs_valid_yaml(self, examples_dir):
        """Перевірка валідності YAML у всіх конфігураціях."""
        for config_file in examples_dir.glob("*.yaml"):
            with open(config_file) as f:
                try:
                    config = yaml.safe_load(f)
                    assert config is not None, f"{config_file.name} порожній"
                except yaml.YAMLError as e:
                    pytest.fail(f"Невалідний YAML у {config_file.name}: {e}")

    @pytest.mark.parametrize("config_name", [
        "config-conservative.yaml",
        "config-aggressive.yaml",
        "config-multi-exchange.yaml",
        "config-ml-focused.yaml",
    ])
    def test_config_has_required_sections(self, examples_dir, config_name):
        """Перевірка наявності обов'язкових секцій."""
        config_path = examples_dir / config_name
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "trading" in config, f"{config_name}: відсутня секція 'trading'"
        assert "strategies" in config, f"{config_name}: відсутня секція 'strategies'"

    @pytest.mark.parametrize("config_name", [
        "config-conservative.yaml",
        "config-aggressive.yaml",
        "config-multi-exchange.yaml",
        "config-ml-focused.yaml",
    ])
    def test_config_trading_section(self, examples_dir, config_name):
        """Перевірка секції trading."""
        config_path = examples_dir / config_name
        with open(config_path) as f:
            config = yaml.safe_load(f)

        trading = config["trading"]

        # Перевірка symbols
        assert "symbols" in trading, f"{config_name}: відсутні symbols"
        assert isinstance(trading["symbols"], list), f"{config_name}: symbols має бути списком"
        assert len(trading["symbols"]) > 0, f"{config_name}: symbols порожній"

        # Перевірка leverage
        if "leverage" in trading:
            leverage = trading["leverage"]
            assert isinstance(leverage, int), f"{config_name}: leverage має бути цілим числом"
            assert 1 <= leverage <= 125, f"{config_name}: leverage поза межами (1-125)"

        # Перевірка paper_trading
        if "paper_trading" in trading:
            assert isinstance(trading["paper_trading"], bool), \
                f"{config_name}: paper_trading має бути boolean"

    @pytest.mark.parametrize("config_name", [
        "config-conservative.yaml",
        "config-aggressive.yaml",
        "config-ml-focused.yaml",
    ])
    def test_config_risk_section(self, examples_dir, config_name):
        """Перевірка секції risk."""
        config_path = examples_dir / config_name
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "risk" not in config:
            pytest.skip(f"{config_name}: секція risk відсутня")

        risk = config["risk"]

        # Перевірка max_position_size
        if "max_position_size" in risk:
            assert 0 < risk["max_position_size"] <= 1, \
                f"{config_name}: max_position_size має бути (0, 1]"

        # Перевірка max_drawdown
        if "max_drawdown" in risk:
            assert 0 < risk["max_drawdown"] <= 1, \
                f"{config_name}: max_drawdown має бути (0, 1]"

        # Перевірка risk_per_trade
        if "risk_per_trade" in risk:
            assert 0 < risk["risk_per_trade"] <= 0.1, \
                f"{config_name}: risk_per_trade має бути (0, 0.1]"

        # Перевірка max_positions
        if "max_positions" in risk:
            assert risk["max_positions"] >= 1, \
                f"{config_name}: max_positions має бути >= 1"

    def test_conservative_config_values(self, examples_dir):
        """Перевірка консервативних значень."""
        config_path = examples_dir / "config-conservative.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Консервативна конфігурація має мати низький leverage
        assert config["trading"]["leverage"] <= 5, \
            "Консервативна конфігурація має мати leverage <= 5"

        # Має бути paper_trading = true
        assert config["trading"]["paper_trading"] is True, \
            "Консервативна конфігурація має мати paper_trading = true"

        # Малий max_position_size
        assert config["risk"]["max_position_size"] <= 0.1, \
            "Консервативна конфігурація має мати max_position_size <= 0.1"

    def test_aggressive_config_values(self, examples_dir):
        """Перевірка агресивних значень."""
        config_path = examples_dir / "config-aggressive.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Агресивна конфігурація має мати високий leverage
        assert config["trading"]["leverage"] >= 10, \
            "Агресивна конфігурація має мати leverage >= 10"

        # Більше symbols
        assert len(config["trading"]["symbols"]) >= 3, \
            "Агресивна конфігурація має мати >= 3 symbols"

    def test_multi_exchange_config(self, examples_dir):
        """Перевірка multi-exchange конфігурації."""
        config_path = examples_dir / "config-multi-exchange.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Має бути секція exchanges
        assert "exchanges" in config["trading"], \
            "Multi-exchange конфігурація має мати секцію exchanges"

        exchanges = config["trading"]["exchanges"]
        assert len(exchanges) >= 2, \
            "Multi-exchange конфігурація має мати >= 2 біржі"

        # Перевірка наявності arbitrage стратегії
        assert config["strategies"].get("arbitrage", {}).get("enabled", False), \
            "Multi-exchange конфігурація має мати увімкнену arbitrage стратегію"

    def test_ml_focused_config(self, examples_dir):
        """Перевірка ML-focused конфігурації."""
        config_path = examples_dir / "config-ml-focused.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Має бути увімкнена ml_signals стратегія
        ml_signals = config["strategies"].get("ml_signals", {})
        assert ml_signals.get("enabled", False), \
            "ML-focused конфігурація має мати увімкнену ml_signals стратегію"

        # Має бути секція model
        assert "model" in ml_signals, \
            "ML-focused конфігурація має мати секцію model"

        # Має бути секція features
        assert "features" in ml_signals, \
            "ML-focused конфігурація має мати секцію features"

    def test_at_least_one_strategy_enabled(self, examples_dir):
        """Перевірка що хоча б одна стратегія увімкнена."""
        for config_file in examples_dir.glob("*.yaml"):
            with open(config_file) as f:
                config = yaml.safe_load(f)

            strategies = config.get("strategies", {})
            enabled_count = sum(
                1 for s in strategies.values()
                if isinstance(s, dict) and s.get("enabled", False)
            )

            assert enabled_count >= 1, \
                f"{config_file.name}: має бути хоча б одна увімкнена стратегія"


# =============================================================================
# Grafana Dashboards Tests
# =============================================================================

class TestGrafanaDashboards:
    """Тести для Grafana дашбордів."""

    def test_grafana_directory_exists(self, grafana_dir):
        """Перевірка існування директорії grafana."""
        assert grafana_dir.exists(), "Директорія grafana не знайдена"

    def test_dashboards_directory_exists(self, grafana_dir):
        """Перевірка існування директорії dashboards."""
        dashboards_dir = grafana_dir / "dashboards"
        assert dashboards_dir.exists(), "Директорія grafana/dashboards не знайдена"

    def test_all_dashboards_exist(self, grafana_dir):
        """Перевірка наявності всіх дашбордів."""
        dashboards_dir = grafana_dir / "dashboards"
        expected_dashboards = [
            "trading-overview.json",
            "risk-management.json",
            "exchange-health.json",
            "strategy-performance.json",
        ]

        for dashboard_name in expected_dashboards:
            dashboard_path = dashboards_dir / dashboard_name
            assert dashboard_path.exists(), f"Дашборд {dashboard_name} не знайдено"

    def test_dashboards_valid_json(self, grafana_dir):
        """Перевірка валідності JSON у всіх дашбордах."""
        dashboards_dir = grafana_dir / "dashboards"

        for dashboard_file in dashboards_dir.glob("*.json"):
            with open(dashboard_file) as f:
                try:
                    dashboard = json.load(f)
                    assert dashboard is not None, f"{dashboard_file.name} порожній"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Невалідний JSON у {dashboard_file.name}: {e}")

    @pytest.mark.parametrize("dashboard_name", [
        "trading-overview.json",
        "risk-management.json",
        "exchange-health.json",
        "strategy-performance.json",
    ])
    def test_dashboard_has_required_fields(self, grafana_dir, dashboard_name):
        """Перевірка наявності обов'язкових полів у дашборді."""
        dashboard_path = grafana_dir / "dashboards" / dashboard_name
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        # Обов'язкові поля для Grafana дашборду
        required_fields = ["title", "uid", "panels", "schemaVersion"]

        for field in required_fields:
            assert field in dashboard, \
                f"{dashboard_name}: відсутнє поле '{field}'"

    @pytest.mark.parametrize("dashboard_name", [
        "trading-overview.json",
        "risk-management.json",
        "exchange-health.json",
        "strategy-performance.json",
    ])
    def test_dashboard_has_panels(self, grafana_dir, dashboard_name):
        """Перевірка наявності панелей у дашборді."""
        dashboard_path = grafana_dir / "dashboards" / dashboard_name
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        panels = dashboard.get("panels", [])
        assert len(panels) > 0, f"{dashboard_name}: дашборд не має панелей"

    @pytest.mark.parametrize("dashboard_name", [
        "trading-overview.json",
        "risk-management.json",
        "exchange-health.json",
        "strategy-performance.json",
    ])
    def test_dashboard_has_templating(self, grafana_dir, dashboard_name):
        """Перевірка наявності змінних (templating)."""
        dashboard_path = grafana_dir / "dashboards" / dashboard_name
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        templating = dashboard.get("templating", {})
        variables = templating.get("list", [])

        # Кожен дашборд має мати datasource змінну
        datasource_vars = [v for v in variables if v.get("type") == "datasource"]
        assert len(datasource_vars) >= 1, \
            f"{dashboard_name}: відсутня datasource змінна"

    @pytest.mark.parametrize("dashboard_name", [
        "trading-overview.json",
        "risk-management.json",
        "exchange-health.json",
        "strategy-performance.json",
    ])
    def test_dashboard_panels_have_targets(self, grafana_dir, dashboard_name):
        """Перевірка що панелі мають targets (запити)."""
        dashboard_path = grafana_dir / "dashboards" / dashboard_name
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        for panel in dashboard.get("panels", []):
            # Пропускаємо row панелі
            if panel.get("type") == "row":
                continue

            targets = panel.get("targets", [])
            # Деякі панелі можуть не мати targets (наприклад, text panels)
            if panel.get("type") not in ["text", "news"]:
                assert len(targets) >= 0, \
                    f"{dashboard_name}: панель '{panel.get('title')}' не має targets"

    @pytest.mark.parametrize("dashboard_name", [
        "trading-overview.json",
        "risk-management.json",
        "exchange-health.json",
        "strategy-performance.json",
    ])
    def test_dashboard_has_unique_uid(self, grafana_dir, dashboard_name):
        """Перевірка унікальності UID дашборду."""
        dashboards_dir = grafana_dir / "dashboards"
        dashboard_path = dashboards_dir / dashboard_name

        with open(dashboard_path) as f:
            dashboard = json.load(f)

        uid = dashboard.get("uid")
        assert uid is not None, f"{dashboard_name}: відсутній UID"
        assert isinstance(uid, str), f"{dashboard_name}: UID має бути рядком"
        assert len(uid) > 0, f"{dashboard_name}: UID порожній"

    def test_all_dashboards_have_unique_uids(self, grafana_dir):
        """Перевірка унікальності UID серед всіх дашбордів."""
        dashboards_dir = grafana_dir / "dashboards"
        uids = []

        for dashboard_file in dashboards_dir.glob("*.json"):
            with open(dashboard_file) as f:
                dashboard = json.load(f)
                uid = dashboard.get("uid")
                if uid:
                    uids.append((dashboard_file.name, uid))

        # Перевірка унікальності
        seen_uids = {}
        for filename, uid in uids:
            if uid in seen_uids:
                pytest.fail(
                    f"Дублікат UID '{uid}': {filename} та {seen_uids[uid]}"
                )
            seen_uids[uid] = filename

    @pytest.mark.parametrize("dashboard_name", [
        "trading-overview.json",
        "risk-management.json",
        "exchange-health.json",
        "strategy-performance.json",
    ])
    def test_dashboard_has_crypto_scalper_tag(self, grafana_dir, dashboard_name):
        """Перевірка наявності тегу crypto-scalper."""
        dashboard_path = grafana_dir / "dashboards" / dashboard_name
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        tags = dashboard.get("tags", [])
        assert "crypto-scalper" in tags, \
            f"{dashboard_name}: відсутній тег 'crypto-scalper'"

    def test_dashboard_refresh_intervals(self, grafana_dir):
        """Перевірка інтервалів оновлення."""
        dashboards_dir = grafana_dir / "dashboards"

        for dashboard_file in dashboards_dir.glob("*.json"):
            with open(dashboard_file) as f:
                dashboard = json.load(f)

            refresh = dashboard.get("refresh")
            # Якщо refresh вказано, перевіряємо формат
            if refresh:
                assert isinstance(refresh, str), \
                    f"{dashboard_file.name}: refresh має бути рядком"


# =============================================================================
# Grafana Provisioning Tests
# =============================================================================

class TestGrafanaProvisioning:
    """Тести для Grafana provisioning конфігурацій."""

    def test_provisioning_directory_exists(self, grafana_dir):
        """Перевірка існування директорії provisioning."""
        provisioning_dir = grafana_dir / "provisioning"
        assert provisioning_dir.exists(), \
            "Директорія grafana/provisioning не знайдена"

    def test_dashboards_provisioning_exists(self, grafana_dir):
        """Перевірка наявності dashboards.yml."""
        dashboards_yml = grafana_dir / "provisioning" / "dashboards.yml"
        assert dashboards_yml.exists(), \
            "Файл grafana/provisioning/dashboards.yml не знайдено"

    def test_datasources_provisioning_exists(self, grafana_dir):
        """Перевірка наявності datasources.yml."""
        datasources_yml = grafana_dir / "provisioning" / "datasources.yml"
        assert datasources_yml.exists(), \
            "Файл grafana/provisioning/datasources.yml не знайдено"

    def test_dashboards_provisioning_valid_yaml(self, grafana_dir):
        """Перевірка валідності dashboards.yml."""
        dashboards_yml = grafana_dir / "provisioning" / "dashboards.yml"
        with open(dashboards_yml) as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "dashboards.yml порожній"
                assert "providers" in config, "dashboards.yml: відсутня секція providers"
            except yaml.YAMLError as e:
                pytest.fail(f"Невалідний YAML у dashboards.yml: {e}")

    def test_datasources_provisioning_valid_yaml(self, grafana_dir):
        """Перевірка валідності datasources.yml."""
        datasources_yml = grafana_dir / "provisioning" / "datasources.yml"
        with open(datasources_yml) as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "datasources.yml порожній"
                assert "datasources" in config, \
                    "datasources.yml: відсутня секція datasources"
            except yaml.YAMLError as e:
                pytest.fail(f"Невалідний YAML у datasources.yml: {e}")

    def test_datasources_has_prometheus(self, grafana_dir):
        """Перевірка наявності Prometheus datasource."""
        datasources_yml = grafana_dir / "provisioning" / "datasources.yml"
        with open(datasources_yml) as f:
            config = yaml.safe_load(f)

        datasources = config.get("datasources", [])
        prometheus_sources = [
            ds for ds in datasources
            if ds.get("type") == "prometheus"
        ]

        assert len(prometheus_sources) >= 1, \
            "datasources.yml: відсутній Prometheus datasource"


# =============================================================================
# GitHub Actions Tests
# =============================================================================

class TestGitHubActions:
    """Тести для GitHub Actions workflows."""

    @pytest.fixture
    def workflows_dir(self, project_root) -> Path:
        """Повертає директорію workflows."""
        return project_root / ".github" / "workflows"

    def test_workflows_directory_exists(self, workflows_dir):
        """Перевірка існування директорії workflows."""
        assert workflows_dir.exists(), \
            "Директорія .github/workflows не знайдена"

    def test_ci_workflow_exists(self, workflows_dir):
        """Перевірка наявності CI workflow."""
        ci_files = list(workflows_dir.glob("ci*.yml")) + \
                   list(workflows_dir.glob("ci*.yaml"))
        assert len(ci_files) >= 1, "CI workflow не знайдено"

    def test_cd_workflow_exists(self, workflows_dir):
        """Перевірка наявності CD workflow."""
        cd_path = workflows_dir / "cd.yml"
        assert cd_path.exists(), "CD workflow не знайдено"

    def test_security_workflow_exists(self, workflows_dir):
        """Перевірка наявності Security workflow."""
        security_path = workflows_dir / "security.yml"
        assert security_path.exists(), "Security workflow не знайдено"

    def test_workflows_valid_yaml(self, workflows_dir):
        """Перевірка валідності YAML у workflows."""
        for workflow_file in workflows_dir.glob("*.yml"):
            with open(workflow_file) as f:
                try:
                    config = yaml.safe_load(f)
                    assert config is not None, f"{workflow_file.name} порожній"
                except yaml.YAMLError as e:
                    pytest.fail(f"Невалідний YAML у {workflow_file.name}: {e}")

    def test_workflows_have_required_fields(self, workflows_dir):
        """Перевірка обов'язкових полів у workflows."""
        for workflow_file in workflows_dir.glob("*.yml"):
            with open(workflow_file) as f:
                config = yaml.safe_load(f)

            assert "name" in config, \
                f"{workflow_file.name}: відсутнє поле 'name'"

            # Має бути або 'on', або True (для 'on: push' etc)
            assert "on" in config or True in config, \
                f"{workflow_file.name}: відсутнє поле 'on'"

            assert "jobs" in config, \
                f"{workflow_file.name}: відсутнє поле 'jobs'"


# =============================================================================
# Pre-commit Tests
# =============================================================================

class TestPreCommit:
    """Тести для pre-commit конфігурації."""

    def test_pre_commit_config_exists(self, project_root):
        """Перевірка наявності .pre-commit-config.yaml."""
        config_path = project_root / ".pre-commit-config.yaml"
        assert config_path.exists(), ".pre-commit-config.yaml не знайдено"

    def test_pre_commit_config_valid_yaml(self, project_root):
        """Перевірка валідності YAML."""
        config_path = project_root / ".pre-commit-config.yaml"
        with open(config_path) as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, ".pre-commit-config.yaml порожній"
            except yaml.YAMLError as e:
                pytest.fail(f"Невалідний YAML: {e}")

    def test_pre_commit_has_repos(self, project_root):
        """Перевірка наявності repos."""
        config_path = project_root / ".pre-commit-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "repos" in config, "Відсутня секція 'repos'"
        assert len(config["repos"]) > 0, "Секція 'repos' порожня"

    def test_pre_commit_has_ruff(self, project_root):
        """Перевірка наявності Ruff hook."""
        config_path = project_root / ".pre-commit-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        repos = config.get("repos", [])
        ruff_repos = [r for r in repos if "ruff" in r.get("repo", "")]

        assert len(ruff_repos) >= 1, "Ruff hook не знайдено"
