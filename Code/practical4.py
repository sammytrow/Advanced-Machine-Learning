def list_funct(l):
    return sum(l)


def subtraction(v1, v2):
    return v1 - v2


def higher_or_lower(l):
    return max(l), min(l)

list_num = [3,7,74,2,5]

assert list_funct(list_num) == 91, "this is wrong you GOOF"
print(list_funct(list_num))
n1, n2 = higher_or_lower(list_num)
assert n1 == 74, " wrong way round numbnut"
print(n1, n2)
assert n2 == 2, " i fancy a mcdonalds "
print(subtraction(n1, n2))
assert subtraction(n1, n2) == 72, " do i get a mcdonalds?"

