# See full repository @ https://github.com/niknecht/mephi-computation-systems-in-nuclear-cpp-py.git
# for flake.nix, unit tests, equivalent cpp implementation (soon for the due-sep18 assignment) and more.

import itertools

# A reminder: This slow monstrosity would be a simple oneliner in C++23. (see due-sep18/main.cpp in the aforementioned repository)
#def main():
data = input()
#   std::views::chuck_by(not_a_ws)
# | std::views::transfrorm(size)
# | max_element
max_len = max(
        len(''.join(group)) for isspace,group in itertools.groupby(data, key=str.isspace) if not isspace
        # std::views::join                   std::views::chunk_by space        std::views::filter out spaces
    )
print(max_len)
   # return 0;
