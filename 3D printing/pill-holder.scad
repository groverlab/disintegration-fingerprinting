// cube([30,25, 5]);
height = 30;

$fn=512;
difference() {
    cylinder(d=100, h=height);
    translate([0,5,0]) cylinder(d=100, h=height);
    translate([40,0,0]) cube([50, 100, height*2], center=true);
    translate([-40,0,0]) cube([50, 100, height*2], center=true);
    translate([0,-47.5,27]) cylinder(d=5, h=5);  // servo shaft hole
    translate([0,-47.5,0]) cylinder(d=4, h=26);  // screw access hole
    translate([0,-47.5,0]) cylinder(d=2, h=50);  // screw access hole
    }
