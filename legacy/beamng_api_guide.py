"""
BeamNG Extension Discovery Tool
Finds available extensions and API activation methods
"""

def print_console_commands():
    """Print common console commands to try in BeamNG."""
    
    print("="*60)
    print("BeamNG Console Commands to Try")
    print("="*60)
    
    commands = [
        # Extension discovery
        "extensions.list()",
        "dump(extensions)",
        "for k,v in pairs(extensions) do print(k) end",
        
        # Research/API related
        "extensions.core_research.start()",
        "extensions.gameplay_research.start()",
        "extensions.research_core.start()",
        "extensions.tech.start()",
        "extensions.api.start()",
        
        # Socket/network related
        "extensions.core_socket.start()",
        "extensions.gameplay_socket.start()",
        "extensions.socket.start()",
        
        # Alternative approaches
        "settings.setValue('research.enabled', true)",
        "settings.setValue('api.enabled', true)",
        "core_research.start()",
        "research_core.start()",
        "tech.start()",
        
        # Network/connection
        "network.start()",
        "network.enable()",
        "core_network.start()",
        
        # BeamNGpy specific
        "extensions.core_beamngpy.start()",
        "extensions.beamngpy.start()",
        "beamngpy.start()",
        
        # General help
        "help()",
        "?",
    ]
    
    print("Try these commands in BeamNG console (~ key):")
    print("Copy and paste each line one at a time:\n")
    
    for i, cmd in enumerate(commands, 1):
        print(f"{i:2d}. {cmd}")
    
    print("\n" + "="*60)
    print("IMPORTANT NOTES:")
    print("="*60)
    print("1. Try 'extensions.list()' FIRST to see available extensions")
    print("2. Look for any extension with 'research', 'api', 'socket', or 'beamngpy' in the name")
    print("3. If you see output like 'Research mode started on port 64256', that's success!")
    print("4. Some commands might give errors - that's normal, keep trying others")
    print("5. You can also check BeamNG's main menu for Research/Developer mode")

def print_alternative_solutions():
    """Print alternative solutions if console commands don't work."""
    
    print("\n" + "="*60)
    print("ALTERNATIVE SOLUTIONS")
    print("="*60)
    
    print("If console commands don't work, try:")
    print()
    print("1. CHECK BEAMNG VERSION:")
    print("   - BeamNG.drive vs BeamNG.tech (tech version has more API features)")
    print("   - Version 0.24+ recommended for BeamNGpy")
    print()
    print("2. MAIN MENU OPTIONS:")
    print("   - Look for 'Research', 'Developer', or 'API' mode in main menu")
    print("   - Check Options/Settings for advanced/developer options")
    print()
    print("3. COMMAND LINE LAUNCH:")
    print("   - Try launching BeamNG with: -research or -api flags")
    print("   - Example: BeamNG.drive.exe -research")
    print()
    print("4. BEAMNGPY DOCUMENTATION:")
    print("   - Check: https://beamngpy.readthedocs.io/")
    print("   - Look for version compatibility info")
    print()
    print("5. MANUAL PORT CHECK:")
    print("   - After trying commands, run our troubleshoot_connection.py again")
    print("   - Should show port 64256 as open if successful")

def main():
    print("BeamNG API Activation Guide")
    print_console_commands()
    print_alternative_solutions()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Copy the console commands above")
    print("2. Open BeamNG console with ~ key")
    print("3. Try commands until you see 'Research mode started' or similar")
    print("4. Run: python troubleshoot_connection.py")
    print("5. If port 64256 shows as open, run: python phase1_basic_setup_simple.py")

if __name__ == "__main__":
    main()