# See full repository @ https://github.com/niknecht/mephi-computation-systems-in-nuclear-cpp-py.git
# for flake.nix, unit tests, equivalent cpp implementation (soon for the due-sep18 assignment) and more.

import itertools
from operator import sub

# auto data = std::cin >> input; ...; input | std::views::chunk(2zu); + materialize in-place (the list() thingy)
data = list(itertools.batched([int(x) for x in input().split()], 2))

# auto k_vec = std::pair{(data.back()->first - data.begin()->first), (data.back()->second - data.begin().second)};
k_vec = ((data[-1][0] - data[0][0]), (data[-1][1] - data[0][1]))

# return data | std::views::find_first_of([auto a0 = std::pair{*data.begin()}](std::pair pp){return (pp-a0).second/k_vec.second == (pp-a0).first/k_vec.first}) == data.end()
alligned = all( (tuple(map(sub, datum, data[0])))[0]/k_vec[0] == (tuple(map(sub, datum, data[0])))[1]/k_vec[1] for datum in data)
print(alligned)
