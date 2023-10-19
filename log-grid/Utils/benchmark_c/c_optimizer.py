"""find fastest compiler settings"""
import shutil
from subprocess import call

flags_all = {"-ftree-vectorize", "-maxv"}
# flags_default = ["-c", "-g", "-fpic", "-Wall", "-Wextra", "-O3", "-ffast-math", "-march=native", "-fopenmp", "-DUSE_OMP"]
flags_default = []


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

        setup_str = f"""
CC={comp}
CFLAGS=-c -g -fpic -Wall -Wextra -O3 -ffast-math -march=native -fopenmp -DUSE_OMP {" ".join(flags_c)}
LDFLAGS=-shared -fopenmp
SOURCES=convolver_c.c
OBJECTS=$(SOURCES:.c=.o)
TARGET=convolver_c.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
    $(CC) $(LDFLAGS) $(OBJECTS) -o $@
    rm -f $(OBJECTS) convolver.html convolver.c
    rm -rf build

.c.o:
    $(CC) $(CFLAGS) $< -o $@

clean:
    rm -f $(OBJECTS) $(TARGET) convolver.html convolver.c convolver.*.so
    rm -rf build
""".replace(
            "    ", "	"
        )
        with open("makefile", "w") as f:
            f.write(setup_str)
        call(["make"])

    do_compile()
    print(f"compiled {flags}")
    call(["python", "run_benchmark.py", str(flags) + str(comp)])


def run_benchmarks():
    """run the benchmarks"""
    flags_totest = [f for f in flags_all if f not in flags_default]

    errors = []
    N_default = 5
    for env in ["clang-15"]:  # ["gcc", "gcc-12", "clang-15"]:
        for flags in [[f"DEFAULT_{i}"] for i in range(N_default)] + [[flag] for flag in flags_totest]:
            try:
                benchmark_config(flags, env)
            except Exception as e:
                errors.append((flags, e))
                print(f"fail for {flags}: {e}")

    if errors:
        print("errors:\n" + "\n".join([f"{f}: {e}" for f, e in errors]))


run_benchmarks()
