
class Fraction:
    """  Fraction class  """
    def __init__(self, top, bottom):
        """constructor Definition"""
        self.num = top
        self.denom = bottom

    def show(self):
        print(f"{self.num}/{self.denom}")

    def __add__(self, other_fraction):
        new_num = self.num * other_fraction.denom + \
                  self.denom * other_fraction.num
        new_den = self.denom * other_fraction.denom

        return Fraction(new_num, new_den)


class Stats:
    def __init__(self, nums):
        self.lst = nums

    def sum(self):
        return sum(self.lst)

    def mean(self):
        output = self.sum()/len(self.lst)
        return output

    def median(self):
        srt = sorted(self.lst)
        if len(srt)%2 == 0:
            m1 = srt[len(srt)//2]
            m2 = srt[len(srt)//2 - 1]
            output = (m1 + m2)/2
        else:
            output = srt[len(srt) // 2]

        return output

    def mode(self):
        l1 = []
        srt = sorted(self.lst)
        i = 0
        while i < len(srt):
            l1.append(srt.count(srt[i]))
            i += 1

        d1 = dict(zip(srt, l1))

        d2 = {k for k,v in d1.items()  if v == max(l1)}

        return d2


if __name__ == "__main__":
    # my_frac = Fraction(12,5)
    # f2 = Fraction(10, 4)
    # print(my_frac + f2)
    n1 = Stats([2,6,1,6,34,12,5,2,4,1,6])
    print("Mean of the n1 is: {}".format(Stats.mean(n1)))
    print("Median of the n1 is: {}".format(Stats.median(n1)))
    print("Mode of the n1 is: {}".format(Stats.mode(n1)))




