import unittest
import numpy as np

from decimal import *

def shape(M):
    return len(M),len(M[0])

def transpose(M):
    return [[row[i] for row in M] for i in range(len(M[0]))]

def matxRound(M, decPts=4):
    m, n=shape(M)
    for i in range(m):
        for j in range(n):
            M[i][j]=round(M[i][j],decPts)

def matxMultiply(A, B):
    if len(A[0]) == len(B):
        return [[sum(map(lambda x: x[0]*x[1], zip(i,j)))
                 for j in zip(*B)] for i in A]
    raise ValueError

def augmentMatrix(A, b):
    aug_matrix=[x[:] for x in A]
    for i in range(len(aug_matrix)):
        aug_matrix[i].append(b[i][0])
    return aug_matrix

def swapRows(M, r1, r2):
    M[r1][:],M[r2][:]=M[r2][:],M[r1][:]

def scaleRow(M, r, scale):
    if scale==0:
        raise ValueError
    else:
        M[r]=[x * scale for x in M[r]]

def addScaledRow(M, r1, r2, scale):
    M[r1]=[x * scale+y for x,y in zip(M[r2],M[r1])]

class LinearRegressionTestCase1(unittest.TestCase):
    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'),'Wrong answer')

    def test_matxMultiply(self):

        for _ in range(100):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d))
            mat2 = np.random.randint(low=-5,high=5,size=(d,c))
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1.tolist(),mat2.tolist()))
            self.assertEqual(dotProduct.shape, dp.shape,
                             'Wrong answer, expected shape{}, but got shape{}'.format(dotProduct.shape, dp.shape))
            self.assertTrue((dotProduct == dp).all(),'Wrong answer')

        mat1 = np.random.randint(low=-10,high=10,size=(r,5))
        mat2 = np.random.randint(low=-5,high=5,size=(4,c))
        mat3 = np.random.randint(low=-5,high=5,size=(6,c))
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
            matxMultiply(mat1.tolist(),mat2.tolist())
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
            matxMultiply(mat1.tolist(),mat3.tolist())

    def test_transpose(self):
        for _ in range(100):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r),"Expected shape{}, but got shape{}".format((c,r),t.shape))
            self.assertTrue((matrix.T == t).all(),'Wrong answer')

    def test_augmentMatrix(self):

        for _ in range(50):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))
            Amat = A.tolist()
            bmat = b.tolist()

            Ab = np.array(augmentMatrix(Amat,bmat))
            ab = np.hstack((A,b))

            self.assertTrue(A.tolist() == Amat,"Matrix A shouldn't be modified")
            self.assertEqual(Ab.shape, ab.shape,
                             'Wrong answer, expected shape{}, but got shape{}'.format(ab.shape, Ab.shape))
            self.assertTrue((Ab == ab).all(),'Wrong answer')

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')

    def test_addScaledRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')


if __name__ == '__main__':
    unittest.main()