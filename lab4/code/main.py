from dataclasses import dataclass
from typing import Optional, Callable
import re

def ResolutionProp(kb: set[tuple[str, ...]]) -> list[str]:
    @dataclass
    class Clause:
        parent1: int
        parent1_idx: int
        parent2: int
        parent2_idx: int
        data: tuple[str, ...]
        count: int

        def __str__(self) -> str:
            if len(self.data) == 1:
                return f"({self.data[0]},)"
            return '(' + ", ".join(self.data) + ')'

    def remove_index(clause: Clause, index: int) -> tuple[str, ...]:
        """Remove given index and return new clause data"""
        return clause.data[:index] + clause.data[index + 1:]

    def update_used(used: list[bool], kb_list: list[Clause], kb_index: int) -> list[bool]:
        """Update used state with given clause"""
        clause: Clause = kb_list[kb_index]
        if clause.parent1 != -1:
            used[clause.parent1] = True
            used = update_used(used, kb_list, clause.parent1)

        if clause.parent2 != -1:
            used[clause.parent2] = True
            used = update_used(used, kb_list, clause.parent2)

        return used

    kb_list: list[Clause] = [Clause(-1, -1, -1, -1, each, idx + 1) for (idx, each) in enumerate(kb)]

    success_flag: bool = False
    matched: list[set[int]] = [set() for _ in range(len(kb))]
    while not success_flag:
        new_clause_added_flag: bool = False

        for (clause_idx1, clause1) in enumerate(kb_list[:]):
            # Check end
            if success_flag:
                break

            # Check all clauses after idx1
            for (clause_idx2, clause2) in enumerate(kb_list[clause_idx1 + 1:]):
                clause_idx2 += clause_idx1 + 1

                # Check end
                if success_flag:
                    break
                
                # Check clause matched
                if clause_idx2 in matched[clause_idx1]:
                    continue
                matched[clause_idx1].add(clause_idx2)

                # Check each literal in clause
                for (literal_idx1, each) in enumerate(kb_list[clause_idx1].data):
                    # Target literal to match
                    target: str = each[1:] if (each[0] == '~') else ("~" + each)
                    if target not in clause2.data:
                        continue

                    # Get new clause data
                    literal_idx2: int = clause2.data.index(target)
                    new_clause: tuple[str, ...] = tuple(set(remove_index(clause1, literal_idx1) + remove_index(clause2, literal_idx2)))

                    # Add to kb_list
                    new_clause_object: Clause = Clause(clause_idx1, literal_idx1, clause_idx2, literal_idx2, new_clause, -1)
                    kb_list.append(new_clause_object)

                    # Check end and update flag
                    new_clause_added_flag = True
                    if len(new_clause) == 0:
                        success_flag = True
                        break
        # Check dead loop
        if not success_flag and not new_clause_added_flag:
            raise ValueError("ResolutionProp failed")

    # Calculate used list
    used: list[bool] = [False] * len(kb_list)
    used[-1] = True
    used = update_used(used, kb_list, len(kb_list) - 1)

    # Construct output
    ret: list[str] = list(map(lambda x: f"{x.count} {str(x)}", kb_list[:len(kb)]))
    count: int = len(kb) + 1
    for (idx, each) in enumerate(kb_list[len(kb):]):
        if not used[idx + len(kb)]:
            continue
        each.count = count

        # Get output index
        parent1: Clause = kb_list[each.parent1]
        output_idx1: str = str(parent1.count)
        if len(parent1.data) != 1:
            output_idx1 += chr(ord('a') + each.parent1_idx)

        parent2: Clause = kb_list[each.parent2]
        output_idx2: str = str(parent2.count)
        if len(parent2.data) != 1:
            output_idx2 += chr(ord('a') + each.parent2_idx)
        ret.append(f"{count} R[{output_idx1}, {output_idx2}] = {str(each)}")

        count += 1
    return ret

def MGU(formula1: str, formula2: str) -> dict[str, str]:
    def is_var(input: str) -> bool:
        if input in "ab":
            return False
        return (len(input) == 1) or (len(input) == 2 and input[0] == input[1])
    
    def get_diff(list1: list[str], list2: list[str]) -> list[list[str]]:
        return [[x, y] for (x, y) in zip(list1, list2) if x != y]

    # Get different of formula
    formula_pattern: re.Pattern = re.compile(r"^~?([a-zA-Z]+)\(([^,]+(,[^,]+)*)\)$")
    formula_match1: list[tuple[str, ...]] = formula_pattern.findall(formula1)
    formula_match2: list[tuple[str, ...]] = formula_pattern.findall(formula2)
    if formula_match1[0][0] != formula_match2[0][0]:
        # Different predicate
        return dict()

    vars1: list[str] = formula_match1[0][1].split(",")
    vars2: list[str] = formula_match2[0][1].split(",")
    diff: list[list[str]] = get_diff(vars1, vars2)

    # Remove same predicate
    index: int = 0
    func_pattern: re.Pattern = re.compile(r"^([a-zA-Z]+)\((.+)\)$")
    while index < len(diff):
        func_match1: list[tuple[str, ...]] = func_pattern.findall(diff[index][0])
        func_match2: list[tuple[str, ...]] = func_pattern.findall(diff[index][1])

        if (len(func_match1) == 0 or len(func_match2) == 0) or (func_match1[0][0] != func_match2[0][0]):
            index += 1
            continue

        arg1: str = func_match1[0][1]
        arg2: str = func_match2[0][1]

        if "," in arg1:
            diff.pop(index)
            diff.extend(get_diff(arg1.split(","), arg2.split(",")))
        else:
            diff[index] = [arg1, arg2]

    ret: dict[str, str] = dict()
    index: int = 0
    while len(diff) != 0:
        replaced_flag: bool = False
        while index < len(diff):
            if not is_var(diff[index][0]):
                diff[index].reverse()

            # Require var and item
            if (not is_var(diff[index][0])) or (is_var(diff[index][1])):
                index += 1
                continue

            # Check var in item
            if re.match(f"[(,]{diff[index][0]}[,)]", diff[index][1]) != None:
                index += 1
                continue
            
            replaced_flag = True
            replace: list[str] = diff[index]
            ret[replace[0]] = replace[1]
            # Apply replace and remove same item
            replace_idx: int = 0
            while replace_idx < len(diff):
                diff[replace_idx][0] = diff[replace_idx][0].replace(replace[0], replace[1])
                diff[replace_idx][1] = diff[replace_idx][1].replace(replace[0], replace[1])
                if diff[replace_idx][0] == diff[replace_idx][1]:
                    diff.pop(replace_idx)
                    if replace_idx < index:
                        index -= 1
                else:
                    replace_idx += 1
        if (len(diff) != 0) and (not replaced_flag):
            return dict()

    return ret

def ResolutionPrinciple(kb: set[tuple[str, ...]]) -> list[str]:
    @dataclass
    class Clause:
        parent1: int
        parent1_idx: int
        parent2: int
        parent2_idx: int
        data: tuple[str, ...]
        sigma: dict[str, str]
        count: int

        def __str__(self) -> str:
            if len(self.data) == 1:
                return f"({self.data[0]},)"
            return '(' + ", ".join(self.data) + ')'

        __repr__ = __str__

    def sigma_to_str(sigma: dict[str, str]) -> str:
        if len(sigma) == 0:
            return ""

        output = str()
        for (key, value) in sigma.items():
            output += f",{key}={value}"
        return f"{{{output[1:]}}}"

    def remove_index(clause: Clause, index: int) -> tuple[str, ...]:
        """Remove given index and return new clause data"""
        return clause.data[:index] + clause.data[index + 1:]

    def update_used(used: list[bool], kb_list: list[Clause], kb_index: int) -> list[bool]:
        """Update used state with given clause"""
        clause: Clause = kb_list[kb_index]
        if clause.parent1 != -1:
            used[clause.parent1] = True
            used = update_used(used, kb_list, clause.parent1)

        if clause.parent2 != -1:
            used[clause.parent2] = True
            used = update_used(used, kb_list, clause.parent2)

        return used

    def apply_sigma(sigma: dict[str, str]) -> Callable[[str], str]:
        def try_replace(match: re.Match[str]) -> str:
            if match.group() in sigma.keys():
                return sigma[match.group()]
            return match.group()

        def inner(literal: str) -> str:
            return re.sub(r"\w+", try_replace, literal)
        return inner

    def MGU(formula1: str, formula2: str) -> dict[str, str]:
        def is_var(input: str) -> bool:
            return (len(input) == 1) or (len(input) == 2 and input[0] == input[1])
        
        def get_diff(list1: list[str], list2: list[str]) -> list[list[str]]:
            return [[x, y] for (x, y) in zip(list1, list2) if x != y]

        # Get different of formula
        formula_pattern: re.Pattern = re.compile(r"^~?([a-zA-Z]+)\(([^,]+(,[^,]+)*)\)$")
        formula_match1: list[tuple[str, ...]] = formula_pattern.findall(formula1)
        formula_match2: list[tuple[str, ...]] = formula_pattern.findall(formula2)
        if formula_match1[0][0] != formula_match2[0][0]:
            # Different predicate
            raise ValueError("Failed to run MGU")

        vars1: list[str] = formula_match1[0][1].split(",")
        vars2: list[str] = formula_match2[0][1].split(",")
        diff: list[list[str]] = get_diff(vars1, vars2)

        # Remove same predicate
        index: int = 0
        func_pattern: re.Pattern = re.compile(r"^([a-zA-Z]+)\((.+)\)$")
        while index < len(diff):
            func_match1: list[tuple[str, ...]] = func_pattern.findall(diff[index][0])
            func_match2: list[tuple[str, ...]] = func_pattern.findall(diff[index][1])

            if (len(func_match1) == 0 or len(func_match2) == 0) or (func_match1[0][0] != func_match2[0][0]):
                index += 1
                continue

            arg1: str = func_match1[0][1]
            arg2: str = func_match2[0][1]

            if "," in arg1:
                diff.pop(index)
                diff.extend(get_diff(arg1.split(","), arg2.split(",")))
            else:
                diff[index] = [arg1, arg2]

        ret: dict[str, str] = dict()
        index: int = 0
        while len(diff) != 0:
            replaced_flag: bool = False
            while index < len(diff):
                if not is_var(diff[index][0]):
                    diff[index].reverse()

                # Require var and item
                if (not is_var(diff[index][0])) or (is_var(diff[index][1])):
                    index += 1
                    continue

                # Check var in item
                if re.match(f"[(,]{diff[index][0]}[,)]", diff[index][1]) != None:
                    index += 1
                    continue
                
                replaced_flag = True
                replace: list[str] = diff[index]
                ret[replace[0]] = replace[1]
                # Apply replace and remove same item
                replace_idx: int = 0
                while replace_idx < len(diff):
                    diff[replace_idx][0] = diff[replace_idx][0].replace(replace[0], replace[1])
                    diff[replace_idx][1] = diff[replace_idx][1].replace(replace[0], replace[1])
                    if diff[replace_idx][0] == diff[replace_idx][1]:
                        diff.pop(replace_idx)
                        if replace_idx < index:
                            index -= 1
                    else:
                        replace_idx += 1
            if (len(diff) != 0) and (not replaced_flag):
                raise ValueError("Failed to run MGU")

        return ret



    kb_list: list[Clause] = [Clause(-1, -1, -1, -1, each, dict(), idx + 1) for (idx, each) in enumerate(kb)]

    success_flag: bool = False
    matched: list[set[int]] = [set() for _ in range(len(kb))]
    generated: set[tuple[str, ...]] = kb.copy()
    while not success_flag:
        new_clause_added_flag: bool = False

        for (clause_idx1, clause1) in enumerate(kb_list[:]):
            # Check end
            if success_flag:
                break

            # Check all clauses after idx1
            for (clause_idx2, clause2) in enumerate(kb_list[clause_idx1 + 1:]):
                clause_idx2 += clause_idx1 + 1
                # Check end
                if success_flag:
                    break

                if clause_idx2 in matched[clause_idx1]:
                    continue
                matched[clause_idx1].add(clause_idx2)

                # Check each literal in clause
                for (literal_idx1, literal1) in enumerate(clause1.data):
                    for (literal_idx2, literal2) in enumerate(clause2.data):
                        if not ((literal1[0] == "~") ^ (literal2[0] == "~")):
                            continue

                        # Calculate sigma
                        sigma: dict[str, str]
                        try:
                            sigma = MGU(literal1, literal2)
                        except ValueError:
                            continue

                        # Get new clause data
                        new_clause_raw: tuple[str, ...] = remove_index(clause1, literal_idx1) + remove_index(clause2, literal_idx2)
                        new_clause: tuple[str, ...] = tuple(set(map(apply_sigma(sigma), new_clause_raw)))
                        if new_clause in generated:
                            continue
                        generated.add(new_clause)

                        # Add to kb_list
                        new_clause_object: Clause = Clause(clause_idx1, literal_idx1, clause_idx2, literal_idx2, new_clause, sigma, -1)
                        kb_list.append(new_clause_object)

                        # Update matched dict
                        matched.append(set())

                        # Check end and update flag
                        new_clause_added_flag = True
                        if len(new_clause) == 0:
                            success_flag = True
                            break
        # Check dead loop
        if not success_flag and not new_clause_added_flag:
            raise ValueError("ResolutionProp failed")

    # Calculate used list
    used: list[bool] = [False] * len(kb_list)
    used[-1] = True
    used = update_used(used, kb_list, len(kb_list) - 1)

    # Construct output
    ret: list[str] = list(map(lambda x: f"{x.count} {str(x)}", kb_list[:len(kb)]))
    count: int = len(kb) + 1
    for (idx, each) in enumerate(kb_list[len(kb):]):
        if not used[idx + len(kb)]:
            continue
        each.count = count

        # Get output index
        parent1: Clause = kb_list[each.parent1]
        output_idx1: str = str(parent1.count)
        if len(parent1.data) != 1:
            output_idx1 += chr(ord('a') + each.parent1_idx)

        parent2: Clause = kb_list[each.parent2]
        output_idx2: str = str(parent2.count)
        if len(parent2.data) != 1:
            output_idx2 += chr(ord('a') + each.parent2_idx)

        ret.append(f"{count} R[{output_idx1}, {output_idx2}]{sigma_to_str(each.sigma)} = {str(each)}")
        count += 1
    return ret

def resolution_prop_test():
    print("ResolutionProp test:\n")
    KB: set[tuple[str, ...]] = {('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)}
    list(map(lambda x: print(x), ResolutionProp(KB)))
    print()

def mgu_test():
    print("MGU test:\n")
    input: tuple[str, str] = "P(xx,a)", "P(b,yy)"
    print(f"MGU{input} -> {MGU(*input)}")
    input = "P(a,xx,f(g(yy)))", "P(zz,f(zz),f(uu))"
    print(f"MGU{input} -> {MGU(*input)}\n")

def resolution_principle_test():
    print("ResolutionPrinciple test 1:\n")
    KB: set[tuple[str, ...]] = {
        ("On(tony,mike)",), 
        ("On(mike,john)",), 
        ("Green(tony)",), 
        ("~Green(john)",), 
        ("~On(xx,yy)", "~Green(xx)", "Green(yy)")
    }
    list(map(lambda x: print(x), ResolutionPrinciple(KB)))
    print("\nResolutionPrinciple test 2:\n")
    KB: set[tuple[str, ...]] = {
        ("A(tony)",),
        ("A(mike)",),
        ("A(john)",),
        ("L(tony,rain)",),
        ("L(tony,snow)",),
        ("~A(x)","S(x)","C(x)"),
        ("~C(y)","~L(y,rain)"),
        ("L(z,snow)","~S(z)"),
        ("~L(tony,u)","~L(mike,u)"),
        ("L(tony,v)","L(mike,v)"),
        ("~A(w)","~C(w)","S(w)")
    }
    list(map(lambda x: print(x), ResolutionPrinciple(KB)))
    print()
    KB: set[tuple[str, ...]] = {
        ("Honest(lisi)",),
        ("~Influential(z)", "Know(z, zhangsan)"),
        ("~Know(u, v)", "Friend(u, v)", "~Honest(u)", "~Honest(v)"),
        ("~Friend(m, n)", "Trust(m, n)"),
        ("~Friend(p, q)", "Trust(q, p)"),
        ("Influential(lisi)",),
        ("~Trust(zhangsan, lisi)",),
        ("~Trust(lisi, zhangsan)",),
        ("Honest(zhangsan)",),
    }
    list(map(lambda x: print(x), ResolutionPrinciple(KB)))
    print()

if __name__ == "__main__":
    resolution_prop_test()
    mgu_test()
    resolution_principle_test()
