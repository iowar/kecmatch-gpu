# kecmatch-gpu
Finds matching solidity function signatures using GPU
calculations on the CPU can take hours

gas saving
----------
$ ./kecmatch64_linux -f swapy -a "(uint256,address)" -s 0x00000000
 Searching | thread => 2535 method => swapy135083001085956(uint256,address) method id => 0x00000000
 Searching | thread => 1635 method => swapy438399002156913(uint256,address) method id => 0x00000000
 Searching | thread => 1611 method => swapy438399002156913(uint256,address) method id => 0x00000000
 Searching - thread => 1078 method => swapy547133001989728(uint256,address) method id => 0x00000000
 Searching | thread => 4094 method => swapy216656004549683(uint256,address) method id => 0x00000000
 Searching | thread => 1139 method => swapy519223005692159(uint256,address) method id => 0x00000000
 Searching - thread => 1152 method => swapy519223005692159(uint256,address) method id => 0x00000000
 
+------------+-------------+------------+----------------+
| card       |  find avg.  | test input |  tester        |
+------------+-------------+------------+----------------+
| RTX 3050   |  16.875 sec |  8         |  @iowar        |
| GTX 1660S  |  17.375 sec |  8         |  @atomicwrites |
+------------+-------------+------------+----------------+

