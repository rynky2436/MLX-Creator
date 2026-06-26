// MLX Creator launcher — minimal Cocoa app. Registers with the window server
// (no endless Dock bounce), runs launch.sh (starts server + opens browser),
// and adds a menu-bar (status bar) icon with Open / Quit. Stops server on quit.
#import <Cocoa/Cocoa.h>
#import <mach-o/dyld.h>

static NSTask *gTask = nil;
static NSStatusItem *gStatusItem = nil;

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    // locate launch.sh next to this executable
    char execPath[4096];
    uint32_t sz = sizeof(execPath);
    if (_NSGetExecutablePath(execPath, &sz) == 0) {
        NSString *dir = [[NSString stringWithUTF8String:execPath] stringByDeletingLastPathComponent];
        NSString *script = [dir stringByAppendingPathComponent:@"launch.sh"];
        gTask = [[NSTask alloc] init];
        gTask.launchPath = @"/bin/bash";
        gTask.arguments = @[ script ];
        @try { [gTask launch]; } @catch (NSException *e) { NSLog(@"launch failed: %@", e); }
    }

    // menu-bar status item
    gStatusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSVariableStatusItemLength];
    NSImage *img = nil;
    if (@available(macOS 11.0, *)) {
        img = [NSImage imageWithSystemSymbolName:@"sparkles" accessibilityDescription:@"MLX Creator"];
    }
    if (img) { img.template = YES; gStatusItem.button.image = img; }
    else { gStatusItem.button.title = @"✦"; }
    gStatusItem.button.toolTip = @"MLX Creator";

    NSMenu *menu = [[NSMenu alloc] init];
    NSMenuItem *open = [[NSMenuItem alloc] initWithTitle:@"Open MLX Creator"
                        action:@selector(openUI:) keyEquivalent:@""];
    open.target = self; [menu addItem:open];
    [menu addItem:[NSMenuItem separatorItem]];
    NSMenuItem *quit = [[NSMenuItem alloc] initWithTitle:@"Quit MLX Creator"
                        action:@selector(quitApp:) keyEquivalent:@"q"];
    quit.target = self; [menu addItem:quit];
    gStatusItem.menu = menu;
}

- (void)openUI:(id)sender {
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"http://127.0.0.1:8200"]];
}

- (void)quitApp:(id)sender {
    [NSApp terminate:nil];
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
