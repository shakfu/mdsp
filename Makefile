# Makefile frontend for mdsp
#
# Packaging uses hatchling; hatch_build.py compiles the Mojo extension for
# wheels. For development, `make build` compiles src/mdsp/_core.so in place.

.PHONY: all sync build rebuild test lint lint-check format format-check \
        typecheck qa clean distclean wheel sdist dist check publish-test \
        publish upgrade coverage coverage-html docs release help

MOJO_ROOT := src/mdsp/_mojo
MOJO_SRC := $(shell find $(MOJO_ROOT) -name '*.mojo')
CORE_SO := src/mdsp/_core.so

# Default target
all: build

# Sync environment (initial setup, installs dependencies + package)
sync:
	@uv sync

# Sync the environment and build the Mojo extension if its sources changed
build: sync $(CORE_SO)

$(CORE_SO): $(MOJO_SRC) hatch_build.py
	@uv run python hatch_build.py $@

# Force a rebuild of the Mojo extension
rebuild:
	@rm -f $(CORE_SO)
	@$(MAKE) build

# Run tests (Python, doctests, and Mojo kernel tests)
test: build
	@uv run pytest -v

# Lint with ruff (applies fixes)
lint:
	@uv run ruff check --fix src/ tests/ hatch_build.py

# Lint with ruff (check only, no fixes)
lint-check:
	@uv run ruff check src/ tests/ hatch_build.py

# Format with ruff
format:
	@uv run ruff format src/ tests/ hatch_build.py

# Check formatting without modifying files
format-check:
	@uv run ruff format --check src/ tests/ hatch_build.py

# Type check with mypy
typecheck:
	@uv run mypy src/mdsp hatch_build.py

# Run a full quality assurance check (non-mutating; mirrors CI)
qa: lint-check format-check typecheck test

# Build a platform wheel with bundled Mojo runtime libraries, then retag it
# for PyPI (auditwheel on Linux, delocate on macOS). Output: dist/
wheel:
	@rm -rf dist/raw
	@uv build --wheel -o dist/raw
ifeq ($(shell uname -s),Darwin)
	@uv run delocate-wheel --require-archs arm64 -w dist dist/raw/*.whl
else
	@uv run auditwheel repair --plat manylinux_2_35_$(shell uname -m) -w dist dist/raw/*.whl
endif
	@rm -rf dist/raw

# Build source distribution
sdist:
	@uv build --sdist

# Check distributions with twine
check:
	@uv run twine check dist/*

# Build both wheel and sdist
dist: wheel sdist check

# Publish to TestPyPI
publish-test: check
	@uv run twine upload --repository testpypi dist/*

# Publish to PyPI
publish: check
	@uv run twine upload dist/*

# Upgrade all dependencies
upgrade:
	@uv lock --upgrade
	@uv sync

# Run tests with coverage
coverage: build
	@uv run pytest -v --cov=src/mdsp --cov-report=term-missing

# Generate HTML coverage report
coverage-html: build
	@uv run pytest -v --cov=src/mdsp --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Build documentation (sphinx is fetched on demand; add it to the dev group
# to pin a version)
docs:
	@uv run --with sphinx sphinx-build -b html docs/ docs/_build/html

# Create a release (bump version, tag, push)
release:
	@echo "Current version: $$(grep '^version' pyproject.toml | head -1)"
	@read -p "New version: " version; \
	sed "s/^version = .*/version = \"$$version\"/" pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml; \
	git add pyproject.toml; \
	git commit -m "Bump version to $$version"; \
	git tag -a "v$$version" -m "Release $$version"; \
	echo "Tagged v$$version. Run 'git push && git push --tags' to publish."

# Clean build artifacts
clean:
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf src/*.egg-info/
	@rm -rf .pytest_cache/
	@find src tests -name "*.so" -delete
	@find src tests -name "*.pyd" -delete
	@find src tests -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean everything, including the resolved environment
distclean: clean
	@rm -rf .venv/ .mypy_cache/ .ruff_cache/

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Sync the environment (default)"
	@echo "  sync         - Sync environment (initial setup)"
	@echo "  build        - Sync environment and build the Mojo extension"
	@echo "  rebuild      - Force a rebuild of the Mojo extension"
	@echo "  test         - Build, then run Python, doctest and Mojo tests"
	@echo "  lint         - Lint with ruff (applies fixes)"
	@echo "  lint-check   - Lint with ruff (check only)"
	@echo "  format       - Format with ruff"
	@echo "  format-check - Check formatting without modifying files"
	@echo "  typecheck    - Type check with mypy"
	@echo "  qa           - Run full quality assurance (non-mutating: lint-check, format-check, typecheck, test)"
	@echo "  wheel        - Build a repaired platform wheel into dist/"
	@echo "  sdist        - Build source distribution"
	@echo "  dist         - Build both wheel and sdist"
	@echo "  check        - Check distributions with twine"
	@echo "  publish-test - Publish to TestPyPI"
	@echo "  publish      - Publish to PyPI"
	@echo "  upgrade      - Upgrade all dependencies"
	@echo "  coverage     - Run tests with coverage"
	@echo "  coverage-html - Generate HTML coverage report"
	@echo "  docs         - Build documentation with Sphinx"
	@echo "  release      - Bump version, tag, and prepare release"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  help         - Show this help message"
