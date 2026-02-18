import sys
def solve():
    N = int(input())
    S = input()
    def search(curr, moves):
        if curr == S: 
            return moves
        if len(curr) >= N: 
            return None
        res = search(curr + "M", moves + "M")
        if res: 
            return res
        flipped = "".join("O" if c == "M" else "M" for c in curr)
        return search(flipped + "O", moves + "O")
    print(search("", ""))