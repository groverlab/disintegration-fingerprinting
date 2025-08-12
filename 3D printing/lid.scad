$fn=128;

difference() {
    cylinder(h=7, d=106, center=true);
    translate([0,0,1.5]) cylinder(h=4, d=100, center=true);
    cylinder(h=14, d=85, center=true);
    }
difference() {
    translate([0,0,-2]) intersection() {
        cylinder(h=3, d=85, center=true);
        translate([15,-50,-1.5]) cube([100,100, 3]);
    }
    translate([20,-6,-10]) cube([3, 12, 20]);
    }
