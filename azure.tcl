# Azure theme for a modern looking tkinter interface
# Inspired by Microsoft's Fluent UI

namespace eval ttk::theme::azure {
    variable colors
    array set colors {
        -fg             "#212529"
        -bg             "#ffffff"
        -disabledfg     "#aaaaaa"
        -disabledbg     "#f0f0f0"
        -selectfg       "#ffffff"
        -selectbg       "#4361ee"
        -primary        "#4361ee"
        -primarydark    "#3a56d4"
        -secondary      "#7209b7"
        -accent         "#06d6a0"
        -warning        "#ffd166"
        -danger         "#ef476f"
        -gray           "#6c757d"
        -lightgray      "#e9ecef"
    }

    proc LoadImages {imgdir} {
        variable I
        foreach file [glob -directory $imgdir *.png] {
            set img [file tail [file rootname $file]]
            set I($img) [image create photo -file $file]
        }
    }

    package require Tk 8.6

    # Create the theme
    ttk::style theme create azure -parent default -settings {
        ttk::style configure . \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -troughcolor $colors(-lightgray) \
            -focuscolor $colors(-primary) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -fieldbackground $colors(-bg) \
            -font "Segoe UI 10" \
            -borderwidth 1 \
            -relief flat

        ttk::style map . -foreground [list disabled $colors(-disabledfg)]

        # Button
        ttk::style configure TButton \
            -anchor center \
            -padding {8 4} \
            -background $colors(-primary) \
            -foreground $colors(-selectfg) \
            -font "Segoe UI 10 bold"

        ttk::style map TButton \
            -background [list active $colors(-primarydark) \
                             disabled $colors(-lightgray)] \
            -foreground [list disabled $colors(-disabledfg)] \
            -relief [list {!disabled !pressed} solid \
                          pressed sunken \
                          disabled flat]

        # Secondary Button (Accent style)
        ttk::style configure Accent.TButton \
            -background $colors(-secondary) \
            -foreground $colors(-selectfg)

        ttk::style map Accent.TButton \
            -background [list active "#560bad" \
                             disabled $colors(-lightgray)]

        # Success Button
        ttk::style configure Success.TButton \
            -background $colors(-accent) \
            -foreground $colors(-selectfg)

        ttk::style map Success.TButton \
            -background [list active "#059669" \
                             disabled $colors(-lightgray)]

        # Danger Button
        ttk::style configure Danger.TButton \
            -background $colors(-danger) \
            -foreground $colors(-selectfg)

        ttk::style map Danger.TButton \
            -background [list active "#dc2f58" \
                             disabled $colors(-lightgray)]

        # Outline Button
        ttk::style configure Outline.TButton \
            -background $colors(-bg) \
            -foreground $colors(-primary) \
            -relief solid \
            -borderwidth 1 \
            -bordercolor $colors(-primary)

        ttk::style map Outline.TButton \
            -background [list active $colors(-lightgray) \
                             disabled $colors(-bg)] \
            -foreground [list disabled $colors(-disabledfg)]

        # Checkbutton
        ttk::style configure TCheckbutton \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -indicatorrelief flat \
            -padding 2

        ttk::style map TCheckbutton \
            -background [list active $colors(-lightgray) \
                             disabled $colors(-bg)]

        # Radiobutton
        ttk::style configure TRadiobutton \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -indicatorrelief flat \
            -padding 2

        ttk::style map TRadiobutton \
            -background [list active $colors(-lightgray) \
                             disabled $colors(-bg)]

        # Entry
        ttk::style configure TEntry \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -fieldbackground $colors(-bg) \
            -borderwidth 1 \
            -relief solid \
            -highlightthickness 1 \
            -highlightcolor $colors(-primary) \
            -padding {6 4}

        ttk::style map TEntry \
            -background [list readonly $colors(-lightgray)] \
            -foreground [list readonly $colors(-fg)] \
            -selectbackground [list !focus $colors(-lightgray) \
                                   focus $colors(-primary)] \
            -selectforeground [list !focus $colors(-fg) \
                                   focus $colors(-selectfg)]

        # Combobox
        ttk::style configure TCombobox \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -fieldbackground $colors(-bg) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -padding {6 4}

        ttk::style map TCombobox \
            -background [list active $colors(-bg) \
                             disabled $colors(-disabledbg)]

        # Spinbox
        ttk::style configure TSpinbox \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -fieldbackground $colors(-bg) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -padding {6 4}

        # Notebook
        ttk::style configure TNotebook \
            -background $colors(-bg) \
            -borderwidth 0

        ttk::style configure TNotebook.Tab \
            -background $colors(-lightgray) \
            -foreground $colors(-fg) \
            -padding {10 6} \
            -borderwidth 0

        ttk::style map TNotebook.Tab \
            -background [list selected $colors(-primary) \
                             active $colors(-primarydark)] \
            -foreground [list selected $colors(-selectfg)] \
            -padding [list selected {10 6}]

        # Scrollbar
        ttk::style configure TScrollbar \
            -background $colors(-lightgray) \
            -troughcolor $colors(-bg) \
            -borderwidth 0 \
            -arrowsize 14

        ttk::style map TScrollbar \
            -background [list hover $colors(-gray) \
                             active $colors(-primary)]

        # Progressbar
        ttk::style configure TProgressbar \
            -background $colors(-primary) \
            -troughcolor $colors(-lightgray)

        # Scale
        ttk::style configure TScale \
            -background $colors(-bg) \
            -troughcolor $colors(-lightgray) \
            -sliderwidth 10 \
            -sliderlength 15

        ttk::style map TScale \
            -background [list active $colors(-bg)]

        # Treeview
        ttk::style configure Treeview \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -fieldbackground $colors(-bg) \
            -borderwidth 0

        ttk::style map Treeview \
            -background [list selected $colors(-primary)] \
            -foreground [list selected $colors(-selectfg)]

        ttk::style configure Treeview.Heading \
            -background $colors(-lightgray) \
            -foreground $colors(-fg) \
            -font "Segoe UI 10 bold" \
            -relief flat

        ttk::style map Treeview.Heading \
            -background [list active $colors(-primary)] \
            -foreground [list active $colors(-selectfg)]

        # Labelframe
        ttk::style configure TLabelframe \
            -background $colors(-bg) \
            -borderwidth 1 \
            -relief solid \
            -bordercolor $colors(-lightgray)

        ttk::style configure TLabelframe.Label \
            -background $colors(-bg) \
            -foreground $colors(-primary) \
            -font "Segoe UI 10 bold"

        # Frame
        ttk::style configure TFrame \
            -background $colors(-bg)

        ttk::style configure Card.TFrame \
            -background $colors(-bg) \
            -relief solid \
            -borderwidth 1 \
            -bordercolor $colors(-lightgray)

        # Label
        ttk::style configure TLabel \
            -background $colors(-bg) \
            -foreground $colors(-fg)

        ttk::style configure Title.TLabel \
            -font "Segoe UI 18 bold" \
            -foreground $colors(-primary)

        ttk::style configure Subtitle.TLabel \
            -font "Segoe UI 14 bold" \
            -foreground $colors(-secondary)

        # Separator
        ttk::style configure TSeparator \
            -background $colors(-lightgray)

        # Sizegrip
        ttk::style configure TSizegrip \
            -background $colors(-bg)
    }
}

# Set the theme
proc set_theme {mode} {
    if {$mode eq "dark"} {
        # Dark mode could be implemented here
        ttk::style theme use azure
    } else {
        ttk::style theme use azure
    }
} 