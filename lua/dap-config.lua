-- IRONFORGE Project DAP Configuration
-- This file contains project-specific debugging configurations

local M = {}

-- Python configurations for IRONFORGE
M.python_configs = {
  {
    name = "IRONFORGE: Main Entry Point",
    type = "python",
    request = "launch",
    program = "${workspaceFolder}/ironforge/__main__.py",
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
    },
  },
  {
    name = "IRONFORGE: Core Module",
    type = "python",
    request = "launch",
    module = "ironforge.core",
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
    },
  },
  {
    name = "IRONFORGE: Data Engine",
    type = "python",
    request = "launch",
    module = "ironforge.data_engine",
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
    },
  },
  {
    name = "IRONFORGE: Temporal Discovery",
    type = "python",
    request = "launch",
    module = "ironforge.temporal.discovery",
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
    },
  },
  {
    name = "IRONFORGE: Run Tests",
    type = "python",
    request = "launch",
    module = "pytest",
    args = {
      "tests/",
      "-v",
      "--no-header",
      "--tb=short"
    },
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
    },
  },
  {
    name = "IRONFORGE: Run Specific Test",
    type = "python",
    request = "launch",
    module = "pytest",
    args = function()
      return {
        vim.fn.input("Test path: ", "tests/", "file"),
        "-v",
        "--no-header",
        "--tb=short"
      }
    end,
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
    },
  },
  {
    name = "IRONFORGE: Profile Performance",
    type = "python",
    request = "launch",
    program = "${file}",
    console = "integratedTerminal",
    justMyCode = true,
    python = "python3",
    cwd = "${workspaceFolder}",
    env = {
      PYTHONPATH = "${workspaceFolder}",
      IRONFORGE_PROFILE = "1",
    },
  },
}

-- Function to load IRONFORGE-specific configurations
function M.load_ironforge_configs()
  local dap = require("dap")

  -- Add IRONFORGE configurations to Python
  if dap.configurations.python then
    for _, config in ipairs(M.python_configs) do
      table.insert(dap.configurations.python, config)
    end
  else
    dap.configurations.python = M.python_configs
  end

  print("IRONFORGE DAP configurations loaded!")
end

-- Auto-load configurations when this file is required
if vim.fn.getcwd():match("IRONFORGE") then
  M.load_ironforge_configs()
end

return M