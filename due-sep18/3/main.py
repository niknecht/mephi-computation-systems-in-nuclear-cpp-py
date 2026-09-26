# See full repository @ https://github.com/niknecht/mephi-computation-systems-in-nuclear-cpp-py.git
# for flake.nix, unit tests, equivalent cpp implementation (soon for the due-sep18 assignment) and more.

import itertools

data = list([int(x) for x in input().split(',')])
n = int(input())

# std::views::adjacent_find_n(...) then count, but not in pethone
# kinda like iota + accumulate, weird and ugly, like this whole language
count = sum(1 for i in range(1, len(data)-1) if data[i] == 0 and data[i-1] == 0 and data[i+1] == 0 and bool(i:= i+1))

print(count >= n)
