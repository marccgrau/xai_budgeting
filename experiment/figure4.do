* Clear the current data
clear all

* Create the data
input str20 condition double mean
"AI Present" -3709.509
"AI Absent" -42.556
end

* Create horizontal bar chart for means
graph hbar (asis) mean, over(condition) blabel(bar, format(%9.2f)) ///
    bar(1, bcolor(gs12)) ytitle("Condition") xtitle("Mean not found")

* Display the graph
graph display
