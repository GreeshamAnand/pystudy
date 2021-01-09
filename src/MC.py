
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


if __name__ == "__main__":
    my_frac = Fraction(12,5)
    f2 = Fraction(10, 4)
    print(my_frac + f2)