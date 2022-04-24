#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys
import multiprocessing
import os
import getpass
import inspect
import argparse
import json
import shutil
import grp
import string
import random
import subprocess
import platform
import textwrap
import typing

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = REPO_ROOT / ".ci-py-scripts"
NPROC = multiprocessing.cpu_count()


class col:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_color(color: str, msg: str, bold: bool, **kwargs: Any) -> None:
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        bold_code = col.BOLD if bold else ""
        print(bold_code + color + msg + col.RESET, **kwargs)
    else:
        print(msg, **kwargs)


warnings: List[str] = []


def clean_exit(msg: str) -> None:
    print_color(col.RED, msg, bold=True, file=sys.stderr)

    for warning in warnings:
        print_color(col.YELLOW, warning, bold=False, file=sys.stderr)

    exit(1)


def cmd(commands: List[Any], **kwargs: Any):
    commands = [str(s) for s in commands]
    command_str = " ".join(commands)
    print_color(col.BLUE, command_str, bold=True)
    proc = subprocess.run(commands, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: '{command_str}'")
    return proc


def get_build_dir(name: str) -> str:
    build_dir = REPO_ROOT / f"build-{name}"
    return str(build_dir.relative_to(REPO_ROOT))


def check_docker():
    executable = shutil.which("docker")
    if executable is None:
        clean_exit("'docker' executable not found, install it first (e.g. 'apt install docker.io')")

    if sys.platform == "linux":
        # Check that the user is in the docker group before running
        try:
            group = grp.getgrnam("docker")
            if getpass.getuser() not in group.gr_mem:
                warnings.append(
                    f"Note: User '{getpass.getuser()}' is not in the 'docker' group, either:\n"
                    " * run with 'sudo'\n"
                    " * add user to 'docker': sudo usermod -aG docker $(whoami), then log out and back in",
                )
        except KeyError:
            warnings.append("Note: 'docker' group does not exist")


def check_gpu():
    if not (sys.platform == "linux" and shutil.which("lshw")):
        # Can't check GPU on non-Linux platforms
        return

    # See if we can check if a GPU is present in case of later failures,
    # but don't block on execution since this isn't critical
    try:
        proc = cmd(
            ["lshw", "-json", "-C", "display"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        stdout = proc.stdout.strip().strip(",")
        stdout = json.loads(stdout)
    except (subprocess.CalledProcessError, json.decoder.JSONDecodeError):
        # Do nothing if any step failed
        return

    if isinstance(stdout, dict):
        # Sometimes lshw outputs a single item as a dict instead of a list of
        # dicts, so wrap it up if necessary
        stdout = [stdout]
    if not isinstance(stdout, list):
        return

    vendors = [s.get("vendor", "").lower() for s in stdout]
    if not any("nvidia" in vendor for vendor in vendors):
        warnings.append(
            "nvidia GPU not found in 'lshw', maybe use --cpu flag when running 'docs' command?"
        )


def check_build():
    if (REPO_ROOT / "build").exists():
        warnings.append(
            "Existing build dir found may be interfering with the Docker "
            "build (you may need to remove it)"
        )


def gen_name(s: str) -> str:
    # random 4 letters
    suffix = "".join([random.choice(string.ascii_lowercase) for i in range(5)])
    return f"{s}-{suffix}"


def docker(name: str, image: str, scripts: List[str], env: Dict[str, str], interactive: bool):
    """
    Invoke a set of bash scripts through docker/bash.sh

    name: container name
    image: docker image name
    scripts: list of bash commands to run
    env: environment to set
    """
    check_docker()

    # As sccache is added to these images these can be uncommented
    sccache_images = {
        # "ci_lint",
        "ci_gpu",
        "ci_cpu",
        # "ci_wasm",
        # "ci_i386",
        "ci_qemu",
        "ci_arm",
        "ci_hexagon",
    }

    if image in sccache_images and os.getenv("USE_SCCACHE", "1") == "1":
        scripts = [
            "sccache --start-server",
        ] + scripts
        # Set the C/C++ compiler so CMake picks them up in the build
        env["CC"] = "/opt/sccache/cc"
        env["CXX"] = "/opt/sccache/c++"
        env["SCCACHE_CACHE_SIZE"] = os.getenv("SCCACHE_CACHE_SIZE", "50G")

    docker_bash = REPO_ROOT / "docker" / "bash.sh"

    command = [docker_bash, "--name", name]
    if interactive:
        command.append("-i")
        command.append("-t")
        scripts = ["interact() {", "  bash", "}", "trap interact 0", ""] + scripts

    for key, value in env.items():
        command.append("--env")
        command.append(f"{key}={value}")

    SCRIPT_DIR.mkdir(exist_ok=True)

    script_file = SCRIPT_DIR / f"{name}.sh"
    with open(script_file, "w") as f:
        f.write("set -eux\n\n")
        f.write("\n".join(scripts))
        f.write("\n")

    command += [image, "bash", str(script_file.relative_to(REPO_ROOT))]

    try:
        cmd(command)
    except RuntimeError as e:
        clean_exit(f"Error invoking Docker: {e}")
    except KeyboardInterrupt:
        cmd(["docker", "stop", "--time", "1", name])
    finally:
        if os.getenv("DEBUG", "0") != "1":
            script_file.unlink()


def docs(
    tutorial_pattern: Optional[str] = None,
    full: bool = False,
    cpu: bool = False,
    interactive: bool = False,
    skip_build: bool = False,
) -> None:
    """
    Build the documentation from gallery/ and docs/. By default this builds only
    the Python docs.

    arguments:
    full -- Build all language docs, not just Python
    precheck -- Run Sphinx precheck script
    tutorial-pattern -- Regex for which tutorials to execute when building docs (can also be set via TVM_TUTORIAL_EXEC_PATTERN)
    cpu -- Run with the ci-cpu image and use CMake defaults for building TVM (if no GPUs are available)
    skip_build -- skip build and setup scripts
    interactive -- start a shell after running build / test scripts
    """
    config = "./tests/scripts/task_config_build_gpu.sh"
    build_dir = get_build_dir("gpu")
    if cpu and full:
        clean_exit("--full cannot be used with --cpu")

    extra_setup = []
    image = "ci_gpu"
    if cpu:
        image = "ci_cpu"
        build_dir = get_build_dir("cpu")
        config = " && ".join(
            [
                "mkdir -p build",
                "pushd build",
                "cp ../cmake/config.cmake .",
                # The docs import tvm.micro, so it has to be enabled in the build
                "echo set\(USE_MICRO ON\) >> config.cmake",
                "popd",
            ]
        )

        # These are taken from the ci-gpu image via pip freeze, consult that
        # if there are any changes: https://github.com/apache/tvm/tree/main/docs#native
        requirements = [
            "Sphinx==4.2.0",
            "tlcpack-sphinx-addon==0.2.1",
            "synr==0.5.0",
            "image==1.5.33",
            "git+https://github.com/guberti/sphinx-gallery.git@ipynb-include-bash",
            "sphinx-rtd-theme==1.0.0",
            "matplotlib==3.3.4",
            "commonmark==0.9.1",
            "Pillow==8.3.2",
            "autodocsumm==0.2.7",
            "docutils==0.16",
        ]

        extra_setup = [
            "python3 -m pip install --user " + " ".join(requirements),
        ]
    else:
        check_gpu()
    config = "./tests/scripts/task_config_build_cpu.sh"

    scripts = extra_setup + [
        config + f" {build_dir}",
        f"./tests/scripts/task_build.py --build-dir {build_dir}",
        "python3 -m pip install --user tlcpack-sphinx-addon==0.2.1 synr==0.6.0",
    ]

    if skip_build:
        scripts = []

    scripts.append("./tests/scripts/task_python_docs.sh")

    if tutorial_pattern is None:
        tutorial_pattern = os.getenv("TVM_TUTORIAL_EXEC_PATTERN", ".py" if full else "none")

    env = {
        "TVM_TUTORIAL_EXEC_PATTERN": tutorial_pattern,
        "PYTHON_DOCS_ONLY": "0" if full else "1",
        "IS_LOCAL": "1",
        "TVM_LIBRARY_PATH": str(REPO_ROOT / build_dir),
    }
    check_build()
    docker(name=gen_name("docs"), image=image, scripts=scripts, env=env, interactive=interactive)


def serve_docs(directory: str = "_docs") -> None:
    """
    Serve the docs using Python's http server

    arguments:
    directory -- Directory to serve from
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        clean_exit("Docs have not been built, run 'ci.py docs' first")
    cmd([sys.executable, "-m", "http.server"], cwd=directory_path)


def lint(interactive: bool = False, fix: bool = False) -> None:
    """
    Run CI's Sanity Check step

    arguments:
    interactive -- start a shell after running build / test scripts
    fix -- where possible (currently black and clang-format) edit files in place with formatting fixes
    """
    env = {}
    if fix:
        env["IS_LOCAL"] = "true"
        env["INPLACE_FORMAT"] = "true"

    docker(
        name=gen_name(f"ci-lint"),
        image="ci_lint",
        scripts=["./tests/scripts/task_lint.sh"],
        env=env,
        interactive=interactive,
    )


Option = Tuple[str, List[str]]


def generate_command(
    name: str,
    options: Dict[str, Option],
    help: str,
    precheck: Optional[Callable[[], None]] = None,
):
    """
    Helper to generate CLIs that:
    1. Build a with a config matching a specific CI Docker image (e.g. 'cpu')
    2. Run tests (either a pre-defined set from scripts or manually via invoking
       pytest)
    3. (optional) Drop down into a terminal into the Docker container
    """

    def fn(
        tests: Optional[List[str]], skip_build: bool = False, interactive: bool = False, **kwargs
    ) -> None:
        """
        arguments:
        tests -- pytest test IDs (e.g. tests/python or tests/python/a_file.py::a_test[param=1])
        skip_build -- skip build and setup scripts
        interactive -- start a shell after running build / test scripts
        """
        if precheck is not None:
            precheck()

        if skip_build:
            scripts = []
        else:
            scripts = [
                f"./tests/scripts/task_config_build_{name}.sh {get_build_dir(name)}",
                f"./tests/scripts/task_build.py --build-dir {get_build_dir(name)}",
                # This can be removed once https://github.com/apache/tvm/pull/10257
                # is merged and added to the Docker images
                "python3 -m pip install --user tlcpack-sphinx-addon==0.2.1 synr==0.6.0",
            ]

        # Check that a test suite was not used alongside specific test names
        if any(v for v in kwargs.values()) and tests is not None:
            option_flags = ", ".join([f"--{k}" for k in options.keys()])
            clean_exit(f"{option_flags} cannot be used with --tests")

        if tests is not None:
            scripts.append(f"python3 -m pytest {' '.join(tests)}")

        # Add named test suites
        for option_name, (_, extra_scripts) in options.items():
            if kwargs.get(option_name, False):
                scripts += extra_scripts

        docker(
            name=gen_name(f"ci-{name}"),
            image=f"ci_{name}",
            scripts=scripts,
            env={
                # Need to specify the library path manually or else TVM can't
                # determine which build directory to use (i.e. if there are
                # multiple copies of libtvm.so laying around)
                "TVM_LIBRARY_PATH": str(REPO_ROOT / get_build_dir(name)),
            },
            interactive=interactive,
        )

    fn.__name__ = name

    return fn, options, help


def check_arm_qemu() -> None:
    """
    Check if a machine is ready to run an ARM Docker image
    """
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        # No need to check anything if the machine runs ARM
        return

    binfmt = Path("/proc/sys/fs/binfmt_misc")
    if not binfmt.exists() or len(list(binfmt.glob("qemu-*"))) == 0:
        clean_exit(
            textwrap.dedent(
                """
        You must run a one-time setup to use ARM containers on x86 via QEMU:

            sudo apt install -y sudo apt-get install qemu binfmt-support qemu-user-static
            docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

        See https://www.stereolabs.com/docs/docker/building-arm-container-on-x86/ for details""".strip(
                    "\n"
                )
            )
        )


def cli_name(s: str) -> str:
    return s.replace("_", "-")


def typing_get_origin(annotation):
    if sys.version_info >= (3, 8):
        return typing.get_origin(annotation)
    else:
        return annotation.__origin__


def typing_get_args(annotation):
    if sys.version_info >= (3, 8):
        return typing.get_args(annotation)
    else:
        return annotation.__args__


def is_optional_type(annotation):
    return (
        hasattr(annotation, "__origin__")
        and (typing_get_origin(annotation) == typing.Union)
        and (type(None) in typing_get_args(annotation))
    )


def add_subparser(
    func: Callable,
    subparsers: Any,
    options: Optional[Dict[str, Option]] = None,
    help: Optional[str] = None,
) -> Any:
    """
    Utility function to make it so subparser commands can be defined locally
    as a function rather than directly via argparse and manually dispatched
    out.
    """

    # Each function is intended follow the example for arguments in PEP257, so
    # split apart the function documentation from the arguments
    split = [s.strip() for s in func.__doc__.split("arguments:\n")]
    if len(split) == 1:
        args_help = None
        command_help = split[0]
    else:
        command_help, args_help = split

    if help is not None:
        command_help = help

    # Parse out the help text for each argument if present
    arg_help_texts = {}
    if args_help is not None:
        for line in args_help.split("\n"):
            line = line.strip()
            name, help_text = [t.strip() for t in line.split(" -- ")]
            arg_help_texts[cli_name(name)] = help_text

    subparser = subparsers.add_parser(cli_name(func.__name__), help=command_help)

    seen_prefixes = set()

    # Add each parameter to the subparser
    signature = inspect.signature(func)
    for name, value in signature.parameters.items():
        if name == "kwargs":
            continue

        arg_cli_name = cli_name(name)
        kwargs: Dict[str, Union[str, bool]] = {"help": arg_help_texts[arg_cli_name]}

        is_optional = is_optional_type(value.annotation)
        if is_optional:
            arg_type = typing_get_args(value.annotation)[0]
        else:
            arg_type = value.annotation

        # Grab the default value if present
        has_default = False
        if value.default is not value.empty:
            kwargs["default"] = value.default
            has_default = True

        # Check if it should be a flag
        if arg_type is bool:
            kwargs["action"] = "store_true"
        else:
            kwargs["required"] = not is_optional and not has_default

        if str(arg_type).startswith("typing.List"):
            kwargs["nargs"] = "+"

        if arg_cli_name[0] not in seen_prefixes:
            subparser.add_argument(f"-{arg_cli_name[0]}", f"--{arg_cli_name}", **kwargs)
            seen_prefixes.add(arg_cli_name[0])
        else:
            subparser.add_argument(f"--{arg_cli_name}", **kwargs)

    if options is not None:
        for option_name, (help, _) in options.items():
            option_cli_name = cli_name(option_name)
            if option_cli_name[0] not in seen_prefixes:
                subparser.add_argument(
                    f"-{option_cli_name[0]}", f"--{option_cli_name}", action="store_true", help=help
                )
                seen_prefixes.add(option_cli_name[0])
            else:
                subparser.add_argument(f"--{option_cli_name}", action="store_true", help=help)

    return subparser


generated = [
    generate_command(
        name="gpu",
        help="Run GPU build and test(s)",
        options={
            "topi": ("run topi tests", ["./tests/scripts/task_python_topi.sh"]),
            "unittest": (
                "run unit tests",
                [
                    "./tests/scripts/task_java_unittest.sh",
                    "./tests/scripts/task_python_unittest_gpuonly.sh",
                    "./tests/scripts/task_python_integration_gpuonly.sh",
                ],
            ),
            "frontend": ("run frontend tests", ["./tests/scripts/task_python_frontend.sh"]),
        },
    ),
    generate_command(
        name="cpu",
        help="Run CPU build and test(s)",
        options={
            "integration": (
                "run integration tests",
                ["./tests/scripts/task_python_integration.sh"],
            ),
            "unittest": (
                "run unit tests",
                [
                    "./tests/scripts/task_python_unittest.sh",
                    "./tests/scripts/task_python_vta_fsim.sh",
                    "./tests/scripts/task_python_vta_tsim.sh",
                ],
            ),
            "frontend": ("run frontend tests", ["./tests/scripts/task_python_frontend_cpu.sh"]),
        },
    ),
    generate_command(
        name="i386",
        help="Run i386 build and test(s)",
        options={
            "integration": (
                "run integration tests",
                [
                    "./tests/scripts/task_python_unittest.sh",
                    "./tests/scripts/task_python_integration_i386only.sh",
                ],
            ),
        },
    ),
    generate_command(
        name="wasm",
        help="Run WASM build and test(s)",
        options={"test": ("run WASM tests", ["./tests/scripts/task_web_wasm.sh"])},
    ),
    generate_command(
        name="qemu",
        help="Run QEMU build and test(s)",
        options={
            "test": (
                "run microTVM tests",
                [
                    "./tests/scripts/task_python_microtvm.sh",
                    "./tests/scripts/task_demo_microtvm.sh",
                ],
            )
        },
    ),
    generate_command(
        name="hexagon",
        help="Run Hexagon build and test(s)",
        options={
            "test": (
                "run Hexagon API/Python tests",
                [
                    "./tests/scripts/task_build_hexagon_api.sh",
                    "./tests/scripts/task_python_hexagon.sh",
                ],
            )
        },
    ),
    generate_command(
        name="arm",
        help="Run ARM build and test(s) (native or via QEMU on x86)",
        precheck=check_arm_qemu,
        options={
            "python": (
                "run full Python tests",
                [
                    "./tests/scripts/task_python_unittest.sh",
                    "./tests/scripts/task_python_arm_compute_library.sh",
                ],
            )
        },
    ),
]


def main():
    description = """
    Run CI jobs locally via Docker. This facilitates reproducing CI failures for
    fast iteration. Note that many of the Docker images required are large (the
    CPU and GPU images are both over 25GB) and may take some time to download on first use.
    """
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command")

    commands = {}

    # Add manually defined commands
    for func in [docs, serve_docs, lint]:
        add_subparser(func, subparsers)
        commands[cli_name(func.__name__)] = func

    # Add generated commands
    for func, options, help in generated:
        add_subparser(func, subparsers, options, help)
        commands[cli_name(func.__name__)] = func

    args = parser.parse_args()

    if args.command is None:
        # Command not found in list, error out
        parser.print_help()
        exit(1)

    func = commands[args.command]

    # Extract out the parsed args and invoke the relevant function
    kwargs = {k: getattr(args, k) for k in dir(args) if not k.startswith("_") and k != "command"}
    func(**kwargs)


if __name__ == "__main__":
    main()
