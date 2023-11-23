"""find fastest compiler settings by brute force

``flags_to_test``: a set of flags to test
``flags_default``: default flags used before testing flags
``N_default``: how many times to run only the default flags
``compilers``: set of compilers to use
``N_cycles``: how many convolutions to run per benchmark
``N``: which grid size to choose
``plot_results``: if True, show results

This will test for each compiler:
* ``N_default`` times the default flags
* Once ``flags_default`` + each member of ``flags_to_test`` separately

Individual benchmark configurations are in ``run_benchmark.py``.
"""
import re
import shutil
from subprocess import call

from Utils.benchmark_c.analyze_results import plot_results

flags_to_test = {"-ftree-vectorize", "-maxv", "-march=native"}
flags_default = ["-c", "-g", "-fpic", "-Wall", "-Wextra", "-O3", "-ffast-math", "-fopenmp", "-DUSE_OMP"]
N_default = 5  # Number of times to run just the default flags (to assess variability)*
compilers = {"clang-15", "gcc-11"}
N_cycles = 1e3
N = 13
show_results = True


def benchmark_config(flags: list[str], comp="clang") -> None:
    """compile a config, benchmark it, save the result"""
    flags_c = list(flags)
    for e in flags_c:
        if e.startswith("DEFAULT"):
            flags_c.remove(e)
    flags_c = flags_default + flags_c

    def do_compile():
        """compile the module"""
        shutil.copyfile("../../pyloggrid/LogGrid/convolver_c.c", "convolver_c.c")
        with open("../../pyloggrid/LogGrid/Makefile", "r", encoding="utf8") as f:
            makefile_src = f.read()

        makefile_src = re.sub(r"CC\s*=.*\n", f"CC={comp}\n", makefile_src)
        makefile_src = re.sub(r"\n\s*python setup_.*\.py build_ext --inplace", "", makefile_src)
        makefile_src = re.sub(r"CFLAGS\s?=.*\n", f"CFLAGS={' '.join(flags_c)}\n", makefile_src)

        with open("makefile", "w", encoding="utf8") as f:
            f.write(makefile_src)

        call(["make"])

    do_compile()
    print(f"compiled {flags}")
    call(["python", "run_benchmark.py", str(flags) + str(comp), str(N_cycles), str(N)])


def run_benchmarks():
    """run the benchmarks"""
    flags_to_test_filtered = [f for f in flags_to_test if f not in flags_default]

    errors = []
    for env in compilers:  # ["gcc", "gcc-12", "clang-15"]:
        for flags in [[f"DEFAULT_{i}"] for i in range(N_default)] + [[flag] for flag in flags_to_test_filtered]:
            try:
                benchmark_config(flags, env)
            except Exception as e:
                errors.append((flags, e))
                print(f"fail for {flags}: {e}")

    if errors:
        print("errors:\n" + "\n".join([f"{f}: {e}" for f, e in errors]))


if __name__ == "__main__":
    run_benchmarks()
    if show_results:
        plot_results()
