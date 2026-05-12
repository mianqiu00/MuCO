# === Basic display ===
hide everything
show cartoon, polymer
show sticks, polymer

set stick_radius, 0.18
set cartoon_transparency, 0.2
bg_color white

select n_term, first polymer
select c_term, last polymer

show sticks, n_term or c_term
show spheres, n_term or c_term
set sphere_scale, 0.3

color blue, n_term
color red, c_term

distance nc_bond, n_term and name N, c_term and name C
set dash_color, yellow
set dash_width, 3
set dash_gap, 0.2

orient
zoom
