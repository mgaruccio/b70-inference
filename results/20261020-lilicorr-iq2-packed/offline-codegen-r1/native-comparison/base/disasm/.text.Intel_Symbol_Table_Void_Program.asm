L0:
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}
(W)     mov (16|M0)              r1.0<1>:ud    r0.0<1;1,0>:ud                   {Compacted}
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}
(W)     mov (16|M0)              r127.0<1>:f   r1.0<1;1,0>:f                    {Compacted,A@1}
(W)     send.gtwy (1|M0)         null     r127  null:0  0x0            0x02000010           {EOT,A@1,$0} // wr:1+0, rd:0; end of thread
L80:
(W)     mov (16|M0)              null<1>:ud    0x192CDCAF:ud                             
(W)     mov (16|M0)              null<1>:ud    0xDD93434F:ud                             
(W)     mov (16|M0)              null<1>:ud    0x0:ud                             
(W)     mov (16|M0)              null<1>:ud    0x2:ud                             
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
