OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
u3(0.107809041472285,1.57079632679490,4.71238898038469) q[0];
u3(0.107809041472285,1.57079632679490,4.71238898038469) q[1];
cx q[0],q[1];
u1(-0.107919454522251) q[1];
cx q[0],q[1];
u3(0.107809041472285,1.57079632679490,4.71238898038469) q[2];
cx q[1],q[2];
u1(-0.107919454522251) q[2];
cx q[1],q[2];
u3(0.107809041472285,1.57079632679490,4.71238898038469) q[3];
cx q[2],q[3];
u1(-0.107919454522251) q[3];
cx q[2],q[3];
