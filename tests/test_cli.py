# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
#
# This file is part of glacier-flow-tools.
#
# GLACIER-FLOW-TOOLS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# GLACIER-FLOW-TOOLS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with glacier-flow-tools; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Tests for the console scripts.
"""

import importlib
from importlib.metadata import entry_points

import pytest

SCRIPTS = ["compute_pathlines", "compute_profiles"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_entry_point_resolves(name):
    """The installed console script points at a callable."""
    (entry_point,) = [e for e in entry_points(group="console_scripts") if e.name == name]
    assert callable(entry_point.load())


@pytest.mark.parametrize("name", SCRIPTS)
def test_help(name, monkeypatch, capsys):
    """``--help`` prints usage and exits with status 0."""
    module = importlib.import_module(f"glacier_flow_tools.{name}")
    monkeypatch.setattr("sys.argv", [name, "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out
