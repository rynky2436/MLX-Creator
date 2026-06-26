// Native arm64 stub for MLX Creator.app — execs the sibling launch.sh.
// (A bash script can't be the CFBundleExecutable: macOS sees no arm64 Mach-O
//  slice and prompts for Rosetta. This compiled stub avoids that.)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <mach-o/dyld.h>

int main(int argc, char *argv[]) {
    char exec_path[PATH_MAX];
    uint32_t size = sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &size) != 0) return 1;

    // exec_path = .../Contents/MacOS/MLX Creator  -> dir = .../Contents/MacOS
    char *slash = strrchr(exec_path, '/');
    if (!slash) return 1;
    *slash = '\0';

    char script[PATH_MAX];
    snprintf(script, sizeof(script), "%s/launch.sh", exec_path);
    execl("/bin/bash", "/bin/bash", script, (char *)NULL);
    perror("execl");
    return 1;
}
