import tensorflow as tf
import numpy as np
import sys


def cca_loss(outdim_size, use_all_singular_values, dcca_params):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_cca_objective(y_true, y_pred):
        """
        It is the loss function of CCA as introduced in the original paper.
        """
        y_pred = tf.reshape(y_pred, (-1, y_pred.shape[2]))

        #! to keep matrices positive semi-definite, Hyperparameters
        r1 = dcca_params['obj_r1']
        r2 = dcca_params['obj_r2']
        eps = dcca_params['obj_eps']

        o1 = o2 = int(y_pred.shape[1] // 2)  # splitting to o1 = o2 = 50

        H1 = tf.transpose(y_pred[:, 0:o1])
        H2 = tf.transpose(y_pred[:, o1:o1+o2])
        m = tf.shape(input=H1)[1]

        # normalising along each timestamp (row)
        H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(
            H1, tf.ones([m, m])
        )
        # normalising along each timestamp (row)
        H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(
            H2, tf.ones([m, m])
        )

        SigmaHat12 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
            H1bar, H2bar, transpose_b=True
        )
        SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
            H1bar, H1bar, transpose_b=True
        ) + r1 * tf.eye(H1.shape[0])
        SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
            H2bar, H2bar, transpose_b=True
        ) + r2 * tf.eye(H1.shape[0])

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        # D1 eigenvalues matrix, V1 eigenvector matrix
        [D1, V1] = tf.linalg.eigh(SigmaHat11)
        [D2, V2] = tf.linalg.eigh(SigmaHat22)  # Added to increase stability

        # taking those eigenvalues greater than eps
        posInd1 = tf.compat.v1.where(tf.greater(D1, eps))
        # get eigen values that are larger than eps
        D1 = tf.gather_nd(D1, posInd1)
        V1 = tf.transpose(
            a=tf.nn.embedding_lookup(
                params=tf.transpose(a=V1), ids=tf.squeeze(posInd1))
        )

        posInd2 = tf.compat.v1.where(tf.greater(D2, eps))
        D2 = tf.gather_nd(D2, posInd2)
        V2 = tf.transpose(
            a=tf.nn.embedding_lookup(
                params=tf.transpose(a=V2), ids=tf.squeeze(posInd2))
        )

        SigmaHat11RootInv = tf.matmul(
            tf.matmul(V1, tf.linalg.tensor_diag(D1 ** -0.5)), V1, transpose_b=True
        )
        SigmaHat22RootInv = tf.matmul(
            tf.matmul(V2, tf.linalg.tensor_diag(D2 ** -0.5)), V2, transpose_b=True
        )

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv,
                         SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            corr = tf.sqrt(tf.linalg.trace(
                tf.matmul(Tval, Tval, transpose_a=True)))
        else:
            [U, V] = tf.linalg.eigh(tf.matmul(Tval, Tval, transpose_a=True))
            U = tf.gather_nd(U, tf.compat.v1.where(tf.greater(U, eps)))
            # kk becomes a scalar
            kk = tf.reshape(tf.cast(tf.shape(input=U), tf.int32), [])
            KK = tf.minimum(kk, outdim_size)
            w, _ = tf.nn.top_k(U, k=KK)
            corr = tf.reduce_sum(input_tensor=tf.sqrt(w))

        return -corr

    return inner_cca_objective
