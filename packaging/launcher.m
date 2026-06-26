// MLX Creator launcher — a minimal Cocoa app so it registers with the window
// server (Dock icon stays put, no endless bounce). It runs launch.sh (which
// starts the server + opens the browser) and stops it on quit.
#import <Cocoa/Cocoa.h>
#import <mach-o/dyld.h>

static NSTask *gTask = nil;

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)note {
    char execPath[4096];
    uint32_t sz = sizeof(execPath);
    if (_NSGetExecutablePath(execPath, &sz) != 0) return;
    NSString *dir = [[NSString stringWithUTF8String:execPath] stringByDeletingLastPathComponent];
    NSString *script = [dir stringByAppendingPathComponent:@"launch.sh"];

    gTask = [[NSTask alloc] init];
    gTask.launchPath = @"/bin/bash";
    gTask.arguments = @[ script ];
    @try { [gTask launch]; } @catch (NSException *e) { NSLog(@"launch failed: %@", e); }
}

- (void)applicationWillTerminate:(NSNotification *)note {
    if (gTask && gTask.isRunning) { @try { [gTask terminate]; } @catch (...) {} }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)s { return NO; }
@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        AppDelegate *d = [[AppDelegate alloc] init];
        [app setDelegate:d];
        [app run];
    }
    return 0;
}
