# See full repository @ https://github.com/niknecht/mephi-computation-systems-in-nuclear-cpp-py.git
# for flake.nix, unit tests, equivalent cpp implementation (soon for the due-sep18 assignment) and more.

data = list([int(x) for x in input().split(',')])


peak = next((i for i in range(1, len(data)-1) if data[i] > data[i-1] and data[i] > data[i+1]), None)
if peak is not None:
    print(peak)
elif len(data) == 1 or data[0] > data[1]:
    print(0)
else:
    print(len(data)-1)
