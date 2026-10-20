L0:
(W)     mov (16|M0)              r127.0<1>:ud  0x0:ud                             
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw             
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x60:ud              {I@2}
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}
(W)     mad (1|M0)               r127.0<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0x80:uw              {I@1}
(W)     mov (8|M0)               r3.0<1>:ud    r1.0<1;1,0>:ud                  
(W)     send.ugm (1|M0)          r1       r127  null:0  0xFF000000            0x6229E500           {A@1,$0} // wr:1+0, rd:2; load.ugm.d32x32t.a32.ca.cc.bti[255]
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud              {$0.src}
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}
(W)     send.ugm (1|M0)          r4       r127  null:0  0xFF000000            0x6219D500           {I@1,$1} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     send.ugm (1|M0)          r5       r127  null:0  0xFF040000            0x6219C500           {$2} // wr:1+0, rd:1; load.ugm.d32x8t.a32.ca.cc.bti[255][A+0x40]
(W)     mov (16|M0)              r98.0<1>:ud   r0.0<1;1,0>:ud                   {Compacted}
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}
        sync.nop                             null                             {Compacted,A@1}
        sync.nop                             null                             {$1.dst}
(W)     mov (2|M0)               r1.8<1>:d     r4.0<1;1,0>:d                    {A@1,$0.dst}
(W)     mov (2|M0)               r2.8<1>:d     r4.2<1;1,0>:d                   
(W)     mul (1|M0)               acc0.0<1>:ud  r98.7<0;1,0>:ud   r1.16<0;1,0>:uw  {I@2}
(W)     macl (1|M0)              r126.0<1>:ud  r98.7<0;1,0>:ud   r1.8<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:ud  r98.7<0;1,0>:ud   r1.16<0;1,0>:uw 
(W)     mach (1|M0)              r8.0<1>:d     r98.7<0;1,0>:ud   r1.8<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r98.7<0;1,0>:ud   r1.18<0;1,0>:uw 
(W)     macl (1|M0)              r9.0<1>:d     r98.7<0;1,0>:ud   r1.9<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:ud  r98.7<0;1,0>:ud   r2.16<0;1,0>:uw  {I@7}
(W)     mov (2|M0)               r3.8<1>:d     r4.4<1;1,0>:d                   
(W)     macl (1|M0)              r125.0<1>:ud  r98.7<0;1,0>:ud   r2.8<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:ud  r98.7<0;1,0>:ud   r2.16<0;1,0>:uw 
(W)     mach (1|M0)              r10.0<1>:d    r98.7<0;1,0>:ud   r2.8<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r98.7<0;1,0>:ud   r2.18<0;1,0>:uw 
(W)     macl (1|M0)              r11.0<1>:d    r98.7<0;1,0>:ud   r2.9<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:ud  r98.7<0;1,0>:ud   r3.16<0;1,0>:uw  {I@6}
(W)     macl (1|M0)              r12.0<1>:ud   r98.7<0;1,0>:ud   r3.8<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:ud  r98.7<0;1,0>:ud   r3.16<0;1,0>:uw 
(W)     mach (1|M0)              r13.0<1>:d    r98.7<0;1,0>:ud   r3.8<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r98.7<0;1,0>:ud   r3.18<0;1,0>:uw 
(W)     macl (1|M0)              r14.0<1>:d    r98.7<0;1,0>:ud   r3.9<0;1,0>:d   
(W)     shl (1|M0)               r16.0<1>:q    r98.6<0;1,0>:ud   2:w              
(W)     add (1|M0)               r13.0<1>:d    r13.0<0;1,0>:d    r14.0<0;1,0>:d   {Compacted,I@2}
(W)     mov (2|M0)               r5.3<1>:d     r4.8<1;1,0>:d                    {$2.dst}
(W)     mov (1|M0)               r12.1<1>:d    r13.0<0;1,0>:d                   {Compacted,I@2}
(W)     mov (2|M0)               r7.0<1>:d     r4.12<1;1,0>:d                   {Compacted}
(W)     add (1|M0)               r15.0<1>:q    r12.0<0;1,0>:q    r3.3<0;1,0>:q    {Compacted,I@2}
(W)     mov (2|M0)               r6.0<1>:d     r4.10<1;1,0>:d                   {Compacted}
(W)     add (1|M0)               r18.0<1>:q    r15.0<0;1,0>:q    r16.0<0;1,0>:q   {Compacted,I@2}
(W)     add (1|M0)               r8.0<1>:d     r8.0<0;1,0>:d     r9.0<0;1,0>:d    {Compacted}
(W)     send.ugm (1|M0)          r19      r18  null:0  0x0            0x02108580           {I@2,$3} // wr:1+0, rd:1; load.ugm.d32x1t.a64
(W)     add (1|M0)               r10.0<1>:d    r10.0<0;1,0>:d    r11.0<0;1,0>:d   {Compacted}
(W)     mov (1|M0)               r99.0<1>:f    r8.0<0;1,0>:f                    {Compacted,I@2}
(W)     mov (1|M0)               r99.1<1>:d    r10.0<0;1,0>:d                   {Compacted,I@1}
(W)     mul (1|M0)               acc0.0<1>:ud  r19.0<0;1,0>:ud   r5.6<0;1,0>:uw   {$3.dst}
(W)     macl (1|M0)              r124.0<1>:ud  r19.0<0;1,0>:ud   r5.3<0;1,0>:ud   {Compacted}
(W)     mul (1|M0)               acc0.0<1>:ud  r19.0<0;1,0>:ud   r5.6<0;1,0>:uw  
(W)     asr (1|M0)               r20.0<1>:d    r19.0<0;1,0>:d    31:w               {Compacted}
(W)     mach (1|M0)              r21.0<1>:d    r19.0<0;1,0>:ud   r5.3<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r19.0<0;1,0>:ud   r5.8<0;1,0>:uw  
(W)     macl (1|M0)              r22.0<1>:d    r19.0<0;1,0>:ud   r5.4<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r5.3<0;1,0>:ud    r20.0<0;1,0>:uw  {I@4}
(W)     add (1|M0)               r21.0<1>:d    r21.0<0;1,0>:d    r22.0<0;1,0>:d   {I@2}
(W)     macl (1|M0)              r22.0<1>:d    r5.3<0;1,0>:ud    r20.0<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:ud  r98.6<0;1,0>:ud   r7.0<0;1,0>:uw  
(W)     macl (1|M0)              r123.0<1>:ud  r98.6<0;1,0>:ud   r7.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:ud  r98.6<0;1,0>:ud   r7.0<0;1,0>:uw  
(W)     mach (1|M0)              r23.0<1>:d    r98.6<0;1,0>:ud   r7.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r98.6<0;1,0>:ud   r7.2<0;1,0>:uw  
(W)     macl (1|M0)              r24.0<1>:d    r98.6<0;1,0>:ud   r7.1<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:ud  r98.6<0;1,0>:ud   r6.0<0;1,0>:uw  
(W)     macl (1|M0)              r122.0<1>:ud  r98.6<0;1,0>:ud   r6.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:ud  r98.6<0;1,0>:ud   r6.0<0;1,0>:uw  
(W)     mach (1|M0)              r25.0<1>:d    r98.6<0;1,0>:ud   r6.0<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r98.6<0;1,0>:ud   r6.2<0;1,0>:uw  
(W)     macl (1|M0)              r26.0<1>:d    r98.6<0;1,0>:ud   r6.1<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r98.1<0;1,0>:d    r5.2<0;1,0>:uw  
(W)     macl (1|M0)              r27.0<1>:d    r98.1<0;1,0>:d    r5.1<0;1,0>:d    {Compacted}
(W)     add (1|M0)               r23.0<1>:d    r23.0<0;1,0>:d    r24.0<0;1,0>:d   {Compacted,I@7}
        add (16|M0)              r121.0<1>:d   r27.0<0;1,0>:d    r2.0<1;1,0>:uw   {I@2}
        cmp (16|M0)   (lt)f1.1   null<1>:d     r121.0<1;1,0>:d   r4.7<0;1,0>:d    {I@1}
(W)     add (1|M0)               r25.0<1>:d    r25.0<0;1,0>:d    r26.0<0;1,0>:d   {Compacted}
(W)     add (1|M0)               r99.2<1>:d    r21.0<0;1,0>:d    r22.0<0;1,0>:d   {Compacted}
(W)     mov (1|M0)               r99.3<1>:d    r23.0<0;1,0>:d                  
(W)     mov (1|M0)               r99.4<1>:d    r25.0<0;1,0>:d                   {I@3}
(~f1.1) goto (16|M0)                         L5232                  L5232                
L1168:
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r4.6<0;1,0>:d     0:w              
(W&~f1.0) jmpi                               L1216                                
L1200:
(W)     add (1|M0)               r4.6<1>:d     r4.6<0;1,0>:d     255:w              
L1216:
        shr (16|M0)              r6.0<1>:uw    r1.0<1;1,0>:uw    3:w              
(W)     asr (1|M0)               r99.5<1>:d    r4.6<0;1,0>:d     8:w               {I@2}
        mov (16|M0)              r120.0<1>:d   r6.0<1;1,0>:uw                   {I@2}
        cmp (16|M0)   (gt)f0.1   null<1>:d     r99.5<0;1,0>:d    r120.0<1;1,0>:d  {I@1}
(f0.1)  goto (16|M0)                         L1328                  L1328                
L1296:
        mov (16|M0)              r100.0<1>:f   0.0:f                              
        goto (16|M0)                         L1328                  L4816                
L1328:
        join (16|M0)                         L4816                                
L1344:
        mov (16|M0)              r6.0<4>:uw    r1.0<1;1,0>:uw                  
(W)     mov (1|M0)               r3.8<1>:d     r124.0<0;1,0>:d                 
(W)     mov (1|M0)               r3.9<1>:d     r99.2<0;1,0>:d                  
        shl (16|M0)              r8.0<1>:q     r6.0<4;1,0>:uw    3:w               {A@2}
(W)     mov (1|M0)               r1.8<1>:d     r126.0<0;1,0>:d                 
(W)     mov (1|M0)               r1.9<1>:d     r99.0<0;1,0>:d                  
        and (16|M0)              r12.0<1>:d    r1.0<1;1,0>:uw    7:w              
(W)     add (1|M0)               r126.1<1>:q   r3.4<0;1,0>:q     r3.0<0;1,0>:q    {I@5}
        and (16|M0)              r10.0<2>:d    r8.0<2;1,0>:d     56:w               {I@5}
        mov (16|M0)              r10.1<2>:d    0:w                              
(W)     mov (1|M0)               r5.4<1>:d     r123.0<0;1,0>:d                 
(W)     mov (1|M0)               r5.5<1>:d     r99.3<0;1,0>:d                  
(W)     mul (16|M0)              acc0.0<1>:d   r99.5<0;1,0>:d    r121.0<2;1,0>:uw
(W)     add (1|M0)               r2.4<1>:q     r1.4<0;1,0>:q     r3.1<0;1,0>:q    {I@7}
        mov (16|M0)              r100.0<1>:f   0.0:f                               {Compacted}
        mul (16|M0)              r117.0<2>:d   r12.0<1;1,0>:d    36:w               {I@7}
        add (16|M0)              r101.0<1>:q   r126.1<0;1,0>:q   r10.0<1;1,0>:q   {Compacted,I@6}
        macl (16|M0)             r119.0<1>:d   r99.5<0;1,0>:d    r121.0<1;1,0>:d 
        mov (16|M0)              r117.1<2>:d   0:w                              
(W)     mov (1|M0)               r99.8<1>:d    1:w                              
(W)     mov (1|M0)               r126.1<1>:d   256:w                               {Compacted}
(W)     mov (1|M0)               r125.1<1>:d   127:w                               {Compacted}
(W)     mov (1|M0)               r104.0<1>:d   2048:w                              
(W)     add (1|M0)               r99.3<1>:q    r2.4<0;1,0>:q     r5.2<0;1,0>:q    {I@7}
L1696:
        shl (16|M0)              r6.0<1>:d     r120.0<1;1,0>:d   3:w               {Compacted}
        add (16|M0)              r7.0<1>:d     r119.0<1;1,0>:d   r120.0<1;1,0>:d  {Compacted,I@7}
(W)     mul (16|M0)              acc0.0<1>:ud  r6.0<1;1,0>:ud    0x24:uw              {I@2}
        asr (16|M0)              r8.0<1>:d     r7.0<1;1,0>:d     31:w               {I@2}
        mach (16|M0)             r13.0<1>:d    r6.0<1;1,0>:ud    0x24:ud             
(W)     mul (16|M0)              acc0.0<1>:ud  r7.0<1;1,0>:ud    0x42:uw             
        mul (16|M0)              r14.0<1>:d    r8.0<1;1,0>:d     0x42:uw              {I@3}
        mach (16|M0)             r15.0<1>:d    r7.0<1;1,0>:ud    0x42:ud             
        mul (16|M0)              r9.0<2>:ud    r7.0<1;1,0>:ud    0x42:uw             
        add (16|M0)              r9.1<2>:d     r15.0<1;1,0>:d    r14.0<1;1,0>:d   {I@2}
        add (16|M0)              r20.0<1>:q    r101.0<1;1,0>:q   r9.0<1;1,0>:q    {I@1}
        add (16|M0)              r29.0<1>:q    r20.0<1;1,0>:q    6:d               {I@1}
        send.ugm (16|M0)         r39      r29  null:0  0x0            0x04100580           {I@1,$4} // wr:2+0, rd:1; load.ugm.d32.a64
        add (16|M0)              r31.0<1>:q    r20.0<1;1,0>:q    2:d              
        send.ugm (16|M0)         r40      r31  null:0  0x0            0x04100580           {I@1,$5} // wr:2+0, rd:1; load.ugm.d32.a64
        mul (16|M0)              r11.0<2>:ud   r6.0<1;1,0>:ud    0x24:uw             
        mov (16|M0)              r11.1<2>:d    r13.0<1;1,0>:d                  
        add (16|M0)              r16.0<1>:q    r99.3<0;1,0>:q    r11.0<1;1,0>:q   {Compacted,I@1}
        add (16|M0)              r22.0<1>:q    r126.1<0;1,0>:q   r9.0<1;1,0>:q    {Compacted}
        add (16|M0)              r18.0<1>:q    r16.0<1;1,0>:q    r117.0<1;1,0>:q  {I@2}
        send.ugm (16|M0)         r37      r22  null:0  0x0            0x04100B80           {I@2,$6} // wr:2+0, rd:1; load.ugm.d16u32.a64
        send.ugm (16|M0)         r41      r18  null:0  0x0            0x04100B80           {I@1,$7} // wr:2+0, rd:1; load.ugm.d16u32.a64
        send.ugm (16|M0)         r24      r18  null:0  0x4000            0x04100580           {$8} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x4]
        send.ugm (16|M0)         r26      r18  null:0  0x8000            0x04100580           {$9} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x8]
        send.ugm (16|M0)         r28      r18  null:0  0xC000            0x04100580           {$10} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0xC]
        send.ugm (16|M0)         r34      r18  null:0  0x10000            0x04100580           {$11} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x10]
        send.ugm (16|M0)         r36      r18  null:0  0x14000            0x04100580           {$12} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x14]
        send.ugm (16|M0)         r38      r18  null:0  0x18000            0x04100580           {$13} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x18]
        send.ugm (16|M0)         r42      r18  null:0  0x1C000            0x04100580           {$14} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x1C]
        send.ugm (16|M0)         r44      r18  null:0  0x20000            0x04100580           {$15} // wr:2+0, rd:1; load.ugm.d32.a64.flat[A+0x20]
        add (16|M0)              r120.0<1>:d   r120.0<1;1,0>:d   2:w              
        cmp (16|M0)   (lt)f2.0   null<1>:d     r120.0<1;1,0>:d   r99.5<0;1,0>:d   {I@1}
        mov (16|M0)              r46.0<1>:d    r39.0<2;1,0>:uw                  {$4.dst}
        mov (16|M0)              r47.0<1>:d    r39.1<2;1,0>:uw                 
        bfn.0xF8 (16|M0)         r51.0<1>:ud   r46.0<1;0>:ud     r125.1<0;0>:ud    r104.0<0>:ud     {I@2} // s0&s1|s2
        shl (16|M0)              r52.0<1>:d    r47.0<1;1,0>:d    16:w               {Compacted,I@2}
        mov (16|M0)              r56.0<2>:ud   r51.0<1;1,0>:ud                  {Compacted,I@2}
        shr (16|M0)              acc2.0<1>:ud  r46.0<1;1,0>:ud   7:w              
        or (16|M0)               r61.0<1>:d    r52.0<1;1,0>:d    r46.0<1;1,0>:d   {I@3}
        add (16|M0)              r63.0<1>:q    r4.7<0;1,0>:q     r56.0<2;1,0>:ud  {I@3}
        bfn.0xF8 (16|M0)         r60.0<1>:ud   acc2.0<1;0>:ud    r125.1<0;0>:ud    r104.0<0>:ud     // s0&s1|s2
        send.ugm (16|M0)         r73      r63  null:0  0x0            0x04100980           {I@2,$0} // wr:2+0, rd:1; load.ugm.d8u32.a64
        mov (16|M0)              r67.0<2>:ud   r60.0<1;1,0>:ud                  {I@1}
        shr (16|M0)              acc2.0<1>:ud  r61.0<1;1,0>:ud   14:w              
        bfn.0xF8 (16|M0)         r78.0<1>:ud   acc2.0<1;0>:ud    r125.1<0;0>:ud    r104.0<0>:ud     // s0&s1|s2
        add (16|M0)              r74.0<1>:q    r4.7<0;1,0>:q     r67.0<2;1,0>:ud  {I@3}
        shr (16|M0)              r53.0<1>:ud   r47.0<1;1,0>:ud   5:w              
        mov (16|M0)              r86.0<2>:ud   r78.0<1;1,0>:ud                  {I@3}
        send.ugm (16|M0)         r85      r74  null:0  0x0            0x04100980           {I@3,$1} // wr:2+0, rd:1; load.ugm.d8u32.a64
        bfn.0xF8 (16|M0)         r62.0<1>:ud   r53.0<1;0>:ud     r125.1<0;0>:ud    r104.0<0>:ud     {I@2} // s0&s1|s2
        add (16|M0)              r93.0<1>:q    r4.7<0;1,0>:q     r86.0<2;1,0>:ud  {I@2}
        send.ugm (16|M0)         r10      r93  null:0  0x0            0x04100980           {I@1,$2} // wr:2+0, rd:1; load.ugm.d8u32.a64
        mov (16|M0)              r69.0<2>:ud   r62.0<1;1,0>:ud                 
        shl (16|M0)              r48.0<1>:d    r40.0<4;1,0>:ub   3:w               {$5.dst}
        add (16|M0)              r79.0<1>:q    r4.7<0;1,0>:q     r69.0<2;1,0>:ud  {I@2}
        send.ugm (16|M0)         r89      r79  null:0  0x0            0x04100980           {I@1,$3} // wr:2+0, rd:1; load.ugm.d8u32.a64
        mov (16|M0)              r54.0<2>:ud   r48.0<1;1,0>:ud                 
        shl (16|M0)              r49.0<1>:d    r40.1<4;1,0>:ub   3:w              
        add (16|M0)              r65.0<1>:q    r4.7<0;1,0>:q     r54.0<2;1,0>:ud  {I@2}
        send.ugm (16|M0)         r76      r65  null:0  0x0            0x04200780           {I@1,$4} // wr:2+0, rd:2; load.ugm.d64.a64
        mov (16|M0)              r58.0<2>:ud   r49.0<1;1,0>:ud                 
        shl (16|M0)              r50.0<1>:d    r40.2<4;1,0>:ub   3:w              
        add (16|M0)              r71.0<1>:q    r4.7<0;1,0>:q     r58.0<2;1,0>:ud  {I@2}
        send.ugm (16|M0)         r81      r71  null:0  0x0            0x04200780           {I@1,$5} // wr:2+0, rd:2; load.ugm.d64.a64
        mov (16|M0)              r83.0<2>:ud   r50.0<1;1,0>:ud                 
        shl (16|M0)              r88.0<1>:d    r40.3<4;1,0>:ub   3:w              
        add (16|M0)              r90.0<1>:q    r4.7<0;1,0>:q     r83.0<2;1,0>:ud  {I@2}
        send.ugm (16|M0)         r7       r90  null:0  0x0            0x04200780           {I@1,$6} // wr:2+0, rd:2; load.ugm.d64.a64
        mov (16|M0)              r96.0<2>:ud   r88.0<1;1,0>:ud                 
        add (16|M0)              r11.0<1>:q    r4.7<0;1,0>:q     r96.0<2;1,0>:ud  {I@1}
        send.ugm (16|M0)         r14      r11  null:0  0x0            0x04200780           {I@1,$7} // wr:2+0, rd:2; load.ugm.d64.a64
        mov (16|M0)              r16.0<1>:b    r73.0<4;1,0>:b                   {$0.dst}
        mov (16|M0)              r23.0<1>:w    r16.0<1;1,0>:b                   {@1,$6.src}
        shr (16|M0)              r92.0<1>:ud   r47.0<1;1,0>:ud   12:w              
        mov (16|M0)              r45.0<1>:uw   r37.0<2;1,0>:uw                  {$6.dst}
        mov (16|M0)              r31.0<1>:b    r85.0<4;1,0>:b                   {$1.dst}
        and (16|M0)              r23.0<1>:w    r23.0<1;1,0>:w    1:w               {I@4}
        mov (16|M0)              r6.0<1>:uw    r41.0<2;1,0>:uw                  {$7.dst}
        shr (16|M0)              r17.0<1>:ud   r73.0<1;1,0>:ud   4:w              
        mov (16|M0)              r9.0<1>:f     r92.0<1;1,0>:ud                  {I@6}
        mov (16|M0)              r95.0<1>:f    r45.0<1;1,0>:hf                  {I@5}
        mov (16|M0)              r35.0<1>:w    r31.0<1;1,0>:b                   {I@4}
        mov (16|M0)              r41.0<2>:b    r23.0<1;1,0>:w                   {I@4}
        mov (16|M0)              r13.0<1>:f    r6.0<1;1,0>:hf                   {I@4}
        shl (16|M0)              r30.0<1>:d    r17.0<1;1,0>:d    7:w               {Compacted,I@3}
        add (16|M0)              acc2.0<1>:f   r9.0<1;1,0>:f     0.5:f               {Compacted,F@3}
        sync.nop                             null                             {F@3}
        mov (16|M0)              r45.0<1>:b    r10.0<4;1,0>:b                   {$2.dst}
        shl (16|M0)              r18.0<1>:d    r73.0<1;1,0>:ud   7:w               {$15.src}
        shl (16|M0)              r19.0<1>:d    r73.0<1;1,0>:ud   14:w              
        shl (16|M0)              r22.0<1>:d    r73.0<1;1,0>:ud   21:w              
        and (16|M0)              r35.0<1>:w    r35.0<1;1,0>:w    1:w               {I@7}
        mov (16|M0)              r47.0<1>:d    r41.0<2;1,0>:ub                  {I@7}
        mul (16|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r95.0<1;1,0>:f  
        shr (16|M0)              r43.0<1>:ud   r85.0<1;1,0>:ud   4:w              
        shl (16|M0)              r33.0<1>:d    r17.0<1;1,0>:d    14:w               {Compacted}
        shl (16|M0)              r40.0<1>:d    r17.0<1;1,0>:d    21:w               {Compacted}
        and (16|M0)              r30.0<1>:d    r30.0<1;1,0>:d    256:w               {Compacted,I@7}
        mov (16|M0)              r49.0<1>:w    r45.0<1;1,0>:b                   {I@7}
        mul (16|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r13.0<1;1,0>:f   {Compacted,F@3}
        and (16|M0)              r19.0<1>:d    r19.0<1;1,0>:d    65536:d               {I@7}
        and (16|M0)              r22.0<1>:d    r22.0<1;1,0>:d    16777216:d               {I@7}
        mov (16|M0)              r48.0<2>:b    r35.0<1;1,0>:w                   {I@7}
        mul (16|M0)              r9.0<1>:f     acc2.0<1;1,0>:f   0.25:f              
        bfn.0xEA (16|M0)         acc2.0<1>:ud  r47.0<1;0>:ud     r18.0<1;0>:ud     r126.1<0>:ud     {A@1} // s0|s1&s2
        mov (16|M0)              r57.0<1>:b    r89.0<4;1,0>:b                   {$3.dst}
        shl (16|M0)              r56.0<1>:d    r43.0<1;1,0>:d    7:w               {I@7}
        and (16|M0)              r33.0<1>:d    r33.0<1;1,0>:d    65536:d               {I@7}
        and (16|M0)              r40.0<1>:d    r40.0<1;1,0>:d    16777216:d               {I@7}
        bfn.0xF8 (16|M0)         r30.0<1>:ud   r17.0<1;0>:ud     r99.8<0;0>:ud     r30.0<1>:ud      {I@7} // s0&s1|s2
        and (16|M0)              r49.0<1>:w    r49.0<1;1,0>:w    1:w               {I@7}
        shl (16|M0)              r50.0<1>:d    r85.0<1;1,0>:ud   7:w              
        shl (16|M0)              r51.0<1>:d    r85.0<1;1,0>:ud   14:w              
        shl (16|M0)              r54.0<1>:d    r85.0<1;1,0>:ud   21:w              
        mov (16|M0)              r55.0<1>:d    r48.0<2;1,0>:ub                  {I@7}
        bfn.0xFE (16|M0)         r22.0<1>:ud   acc2.0<1;0>:ud    r19.0<1;0>:ud     r22.0<1>:ud      // s0|s1|s2
        shl (16|M0)              r59.0<1>:d    r43.0<1;1,0>:d    14:w               {Compacted}
        shl (16|M0)              r62.0<1>:d    r43.0<1;1,0>:d    21:w               {Compacted}
        mov (16|M0)              r65.0<1>:w    r57.0<1;1,0>:b                   {$4.src}
        and (16|M0)              r56.0<1>:d    r56.0<1;1,0>:d    256:w              
        bfn.0xFE (16|M0)         r40.0<1>:ud   r30.0<1;0>:ud     r33.0<1;0>:ud     r40.0<1>:ud      {I@7} // s0|s1|s2
        mov (16|M0)              r64.0<2>:b    r49.0<1;1,0>:w                   {I@7}
        shr (16|M0)              r63.0<1>:ud   r10.0<1;1,0>:ud   4:w              
        and (16|M0)              r51.0<1>:d    r51.0<1;1,0>:d    65536:d               {I@7}
        and (16|M0)              r54.0<1>:d    r54.0<1;1,0>:d    16777216:d               {I@7}
        bfn.0xEA (16|M0)         r55.0<1>:ud   r55.0<1;0>:ud     r50.0<1;0>:ud     r126.1<0>:ud     {I@7} // s0|s1&s2
        mul (16|M0)              acc2.0<1>:d   r22.0<1;1,0>:d    255:w               {I@7}
        shl (16|M0)              r68.0<1>:d    r10.0<1;1,0>:ud   7:w              
        shl (16|M0)              r69.0<1>:d    r10.0<1;1,0>:ud   14:w              
        shl (16|M0)              r72.0<1>:d    r10.0<1;1,0>:ud   21:w               {$5.src}
        and (16|M0)              r59.0<1>:d    r59.0<1;1,0>:d    65536:d              
        and (16|M0)              r62.0<1>:d    r62.0<1;1,0>:d    16777216:d              
        and (16|M0)              r65.0<1>:w    r65.0<1;1,0>:w    1:w              
        bfn.0xF8 (16|M0)         r56.0<1>:ud   r43.0<1;0>:ud     r99.8<0;0>:ud     r56.0<1>:ud      // s0&s1|s2
        mul (16|M0)              r66.0<1>:d    r40.0<1;1,0>:d    255:w              
        mov (16|M0)              r73.0<1>:d    r64.0<2;1,0>:ub                 
        xor (16|M0)              r67.0<1>:d    acc2.0<1;1,0>:d   r76.0<2;1,0>:d   {$4.dst}
        and (16|M0)              acc2.0<1>:d   acc2.0<1;1,0>:d   16843009:d              
        shl (16|M0)              r74.0<1>:d    r63.0<1;1,0>:d    7:w              
        bfn.0xFE (16|M0)         r54.0<1>:ud   r55.0<1;0>:ud     r51.0<1;0>:ud     r54.0<1>:ud      // s0|s1|s2
        and (16|M0)              r69.0<1>:d    r69.0<1;1,0>:d    65536:d              
        and (16|M0)              r72.0<1>:d    r72.0<1;1,0>:d    16777216:d              
        mov (16|M0)              r88.0<2>:b    r65.0<1;1,0>:w                   {I@7}
        bfn.0xFE (16|M0)         r62.0<1>:ud   r56.0<1;0>:ud     r59.0<1;0>:ud     r62.0<1>:ud      {I@7} // s0|s1|s2
        xor (16|M0)              r78.0<1>:d    r66.0<1;1,0>:d    r76.1<2;1,0>:d   {I@7}
        and (16|M0)              r79.0<1>:d    r66.0<1;1,0>:d    16843009:d              
        bfn.0xEA (16|M0)         r73.0<1>:ud   r73.0<1;0>:ud     r68.0<1;0>:ud     r126.1<0>:ud     {I@7} // s0|s1&s2
        add (16|M0)              r75.0<1>:d    r67.0<1;1,0>:d    acc2.0<1;1,0>:d  {I@7}
        shr (16|M0)              r87.0<1>:ud   r89.0<1;1,0>:ud   4:w              
        shl (16|M0)              r83.0<1>:d    r63.0<1;1,0>:d    14:w               {Compacted}
        shl (16|M0)              r86.0<1>:d    r63.0<1;1,0>:d    21:w               {Compacted}
        and (16|M0)              r74.0<1>:d    r74.0<1;1,0>:d    256:w               {Compacted}
        mul (16|M0)              acc2.0<1>:d   r54.0<1;1,0>:d    255:w               {Compacted}
        shl (16|M0)              r11.0<1>:d    r89.0<1;1,0>:ud   14:w               {$7.src}
        shl (16|M0)              r16.0<1>:d    r89.0<1;1,0>:ud   21:w              
        shl (16|M0)              r10.0<1>:d    r89.0<1;1,0>:ud   7:w              
        mov (16|M0)              r17.0<1>:d    r88.0<2;1,0>:ub                 
        mul (16|M0)              r97.0<1>:d    r62.0<1;1,0>:d    255:w               {Compacted}
        add (16|M0)              r93.0<1>:d    r78.0<1;1,0>:d    r79.0<1;1,0>:d   {Compacted}
        bfn.0xFE (16|M0)         r72.0<1>:ud   r73.0<1;0>:ud     r69.0<1;0>:ud     r72.0<1>:ud      // s0|s1|s2
        dp4a (16|M0)             r96.0<1>:d    0:w                r75.0<1;0>:d      r24.0<1>:d       {$8.dst}
        xor (16|M0)              r6.0<1>:d     acc2.0<1;1,0>:d   r81.0<2;1,0>:d   {$5.dst}
        and (16|M0)              acc2.0<1>:d   acc2.0<1;1,0>:d   16843009:d              
        shl (16|M0)              r18.0<1>:d    r87.0<1;1,0>:d    7:w              
        and (16|M0)              r83.0<1>:d    r83.0<1;1,0>:d    65536:d              
        and (16|M0)              r86.0<1>:d    r86.0<1;1,0>:d    16777216:d              
        bfn.0xF8 (16|M0)         r74.0<1>:ud   r63.0<1;0>:ud     r99.8<0;0>:ud     r74.0<1>:ud      // s0&s1|s2
        and (16|M0)              r11.0<1>:d    r11.0<1;1,0>:d    65536:d              
        and (16|M0)              r16.0<1>:d    r16.0<1;1,0>:d    16777216:d              
        bfn.0xEA (16|M0)         r17.0<1>:ud   r17.0<1;0>:ud     r10.0<1;0>:ud     r126.1<0>:ud     // s0|s1&s2
        and (16|M0)              r25.0<1>:d    r97.0<1;1,0>:d    16843009:d              
        dp4a (16|M0)             r19.0<1>:d    r96.0<1;0>:d      r93.0<1;0>:d      r26.0<1>:d       {@7,$9.dst}
        xor (16|M0)              r24.0<1>:d    r97.0<1;1,0>:d    r81.1<2;1,0>:d  
        add (16|M0)              r23.0<1>:d    r6.0<1;1,0>:d     acc2.0<1;1,0>:d 
        mul (16|M0)              r26.0<1>:d    r72.0<1;1,0>:d    255:w               {Compacted}
        shl (16|M0)              r27.0<1>:d    r87.0<1;1,0>:d    14:w               {Compacted}
        shl (16|M0)              r30.0<1>:d    r87.0<1;1,0>:d    21:w               {Compacted}
        and (16|M0)              r18.0<1>:d    r18.0<1;1,0>:d    256:w               {Compacted}
        bfn.0xFE (16|M0)         r86.0<1>:ud   r74.0<1;0>:ud     r83.0<1;0>:ud     r86.0<1>:ud      // s0|s1|s2
        bfn.0xFE (16|M0)         r16.0<1>:ud   r17.0<1;0>:ud     r11.0<1;0>:ud     r16.0<1>:ud      {I@7} // s0|s1|s2
        add (16|M0)              r31.0<1>:d    r24.0<1;1,0>:d    r25.0<1;1,0>:d   {I@7}
        dp4a (16|M0)             r35.0<1>:d    r19.0<1;0>:d      r23.0<1;0>:d      r28.0<1>:d       {@7,$10.dst}
        xor (16|M0)              r37.0<1>:d    r26.0<1;1,0>:d    r7.0<2;1,0>:d    {@7,$6.dst}
        and (16|M0)              r39.0<1>:d    r26.0<1;1,0>:d    16843009:d              
        and (16|M0)              r27.0<1>:d    r27.0<1;1,0>:d    65536:d               {I@7}
        and (16|M0)              r30.0<1>:d    r30.0<1;1,0>:d    16777216:d               {I@7}
        bfn.0xF8 (16|M0)         r18.0<1>:ud   r87.0<1;0>:ud     r99.8<0;0>:ud     r18.0<1>:ud      {I@7} // s0&s1|s2
        mul (16|M0)              acc2.0<1>:d   r86.0<1;1,0>:d    255:w               {Compacted,I@7}
        mul (16|M0)              r47.0<1>:d    r16.0<1;1,0>:d    255:w               {Compacted,I@7}
        dp4a (16|M0)             r40.0<1>:d    r35.0<1;0>:d      r31.0<1;0>:d      r34.0<1>:d       {@7,$11.dst}
        add (16|M0)              r43.0<1>:d    r37.0<1;1,0>:d    r39.0<1;1,0>:d   {I@7}
        xor (16|M0)              r45.0<1>:d    acc2.0<1;1,0>:d   r7.1<2;1,0>:d   
        and (16|M0)              r46.0<1>:d    acc2.0<1;1,0>:d   16843009:d              
        bfn.0xFE (16|M0)         r30.0<1>:ud   r18.0<1;0>:ud     r27.0<1;0>:ud     r30.0<1>:ud      {I@7} // s0|s1|s2
        xor (16|M0)              r49.0<1>:d    r47.0<1;1,0>:d    r14.0<2;1,0>:d   {@6,$7.dst}
        and (16|M0)              r50.0<1>:d    r47.0<1;1,0>:d    16843009:d              
        dp4a (16|M0)             r48.0<1>:d    r40.0<1;0>:d      r43.0<1;0>:d      r36.0<1>:d       {@6,$12.dst}
        add (16|M0)              r45.0<1>:d    r45.0<1;1,0>:d    r46.0<1;1,0>:d   {Compacted,I@5}
        mul (16|M0)              acc2.0<1>:d   r30.0<1;1,0>:d    255:w               {Compacted,I@5}
        add (16|M0)              r55.0<1>:d    r49.0<1;1,0>:d    r50.0<1;1,0>:d   {I@4}
        dp4a (16|M0)             r51.0<1>:d    r48.0<1;0>:d      r45.0<1;0>:d      r38.0<1>:d       {@3,$13.dst}
        xor (16|M0)              r53.0<1>:d    acc2.0<1;1,0>:d   r14.1<2;1,0>:d  
        and (16|M0)              acc2.0<1>:d   acc2.0<1;1,0>:d   16843009:d              
        dp4a (16|M0)             r56.0<1>:d    r51.0<1;0>:d      r55.0<1;0>:d      r42.0<1>:d       {@3,$14.dst}
        add (16|M0)              r53.0<1>:d    r53.0<1;1,0>:d    acc2.0<1;1,0>:d  {I@3}
        dp4a (16|M0)             r57.0<1>:d    r56.0<1;0>:d      r53.0<1;0>:d      r44.0<1>:d       {@1,$15.dst}
        mov (16|M0)              r58.0<1>:f    r57.0<1;1,0>:d                   {I@1}
        mad (16|M0)              r100.0<1>:f   r100.0<1;0>:f     r9.0<1;0>:f       r58.0<1>:f       {F@1}
(f2.0)  goto.b (16|M0)                       L4816                  L1696                
L4816:
        join (16|M0)                         L5232                                
L4832:
(W)     mov (8|M0)               r6.0<1>:ud    r100.8<1;1,0>:ud                 {F@1}
(W)     mov (8|M0)               r6.8<1>:ud    r100.0<1;1,0>:ud                
        cmp (16|M0)   (eq)f0.0   null<2>:w     r1.0<1;1,0>:w     0:w              
        add (16|M0)              r7.0<1>:f     r100.0<1;1,0>:f   r6.0<1;1,0>:f    {Compacted,I@2}
(W)     mov (4|M0)               r8.0<1>:ud    r7.4<1;1,0>:ud                   {Compacted,F@1}
(W)     mov (4|M0)               r8.4<1>:ud    r7.0<1;1,0>:ud                  
(W)     mov (4|M0)               r8.8<1>:ud    r7.12<1;1,0>:ud                 
(W)     mov (4|M0)               r8.12<1>:ud   r7.8<1;1,0>:ud                  
        add (16|M0)              r7.0<1>:f     r7.0<1;1,0>:f     r8.0<1;1,0>:f    {I@1}
(W)     mov (4|M0)               r9.0<4>:ud    r7.2<4;1,0>:ud                   {F@1}
(W)     mov (4|M0)               r9.2<4>:ud    r7.0<4;1,0>:ud                  
(W)     mov (4|M0)               r9.1<4>:ud    r7.3<4;1,0>:ud                  
(W)     mov (4|M0)               r9.3<4>:ud    r7.1<4;1,0>:ud                  
        add (16|M0)              r116.0<1>:f   r7.0<1;1,0>:f     r9.0<1;1,0>:f    {Compacted,I@1}
(W)     mov (8|M0)               r10.0<2>:ud   r116.1<2;1,0>:ud                 {Compacted,F@1}
(W)     mov (8|M0)               r10.1<2>:ud   r116.0<2;1,0>:ud                 {Compacted}
        mov (16|M0)              r103.0<1>:f   r10.0<1;1,0>:f                   {Compacted,I@1}
(~f0.0) goto (16|M0)                         L5232                  L5232                
L5072:
(W)     mov (1|M0)               r1.8<1>:d     r125.0<0;1,0>:d                 
(W)     mov (1|M0)               r1.9<1>:d     r99.1<0;1,0>:d                  
(W)     mov (1|M0)               r3.8<1>:d     r122.0<0;1,0>:d                 
(W)     mov (1|M0)               r3.9<1>:d     r99.4<0;1,0>:d                  
        mov (16|M0)              r7.0<2>:ud    r121.0<1;1,0>:ud                
(W)     add (1|M0)               r2.4<1>:q     r1.4<0;1,0>:q     r3.2<0;1,0>:q    {I@4}
        shl (16|M0)              r9.0<1>:q     r7.0<2;1,0>:d     2:w               {A@1}
(W)     add (1|M0)               r5.2<1>:q     r2.4<0;1,0>:q     r3.4<0;1,0>:q    {I@2}
        add (16|M0)              r6.0<1>:f     r116.0<1;1,0>:f   r103.0<1;1,0>:f  {Compacted}
        add (16|M0)              r11.0<1>:q    r5.2<0;1,0>:q     r9.0<1;1,0>:q    {Compacted,I@1}
        send.ugm (16|M0)         null     r11  r6:1  0x0            0x04000584           {A@1,$0} // wr:2+1, rd:0; store.ugm.d32.a64
L5232:
        join (16|M0)                         L5248                                
L5248:
(W)     mov (16|M0)              r127.0<1>:f   r98.0<1;1,0>:f                  
(W)     send.gtwy (1|M0)         null     r127  null:0  0x0            0x02000010           {EOT,F@1,$1} // wr:1+0, rd:0; end of thread
L5280:
(W)     mov (16|M0)              null<1>:ud    0x24CB2415:ud                             
(W)     mov (16|M0)              null<1>:ud    0x60A62068:ud                             
(W)     mov (16|M0)              null<1>:ud    0x0:ud                             
(W)     mov (16|M0)              null<1>:ud    0x1:ud                             
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
