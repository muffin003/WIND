from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
GUIDES = (ROOT / "web" / "guide-content.js").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_english_is_the_default_handbook_language():
    assert '<html lang="en">' in INDEX
    assert 'data-lang="en" class="active"' in INDEX
    assert 'let language = "en";' in APP
    assert 'localStorage.getItem("wind-language-v2")' in APP
    assert 'localStorage.getItem("wind-language")' not in APP


def test_every_workbench_view_has_a_static_guide():
    views = set(
        re.findall(r'<section class="view(?: active)?" data-view="([^"]+)"', INDEX)
    )
    guides = set(re.findall(r'data-guide-page="([^"]+)"', INDEX))
    assert guides == views
    assert "data-guide-toggle" in APP


def test_handbook_contains_local_start_instructions_in_all_languages():
    assert "Run locally after downloading" in GUIDES
    assert "Локальный запуск после скачивания" in GUIDES
    assert "下载后的本地启动" in GUIDES
    assert "python -m wind_benchmark.web" in GUIDES
    assert "### Start locally after downloading the repository" in README


def test_pages_workflow_publishes_the_static_web_directory():
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "path: web" in workflow
