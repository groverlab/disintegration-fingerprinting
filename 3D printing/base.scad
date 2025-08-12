$fn=120;
difference() {
    cylinder(h=55, d1=72, d2=83);  // outer solid
    translate([0,0,6]) cylinder(h=50, d1=63, d2=72);  // inner cup
    rotate([0,0,-90]) translate([33.5,-4.25,38]) rotate([0,5.0,0]) cube([3, 8.5, 24.5]); // sensor cutout
    }
difference() {
    cube([139, 139, 10], center=true);
    translate([0,0,-5]){ 
        cube([129, 129, 10], center=true);
        cube([139,80,10], center=true);
        cube([80, 139,10], center=true);
        }
    }
translate([0,-60,5]) linear_extrude(1) {
    text("Grover Lab              UC Riverside", 6, halign="center");
    }