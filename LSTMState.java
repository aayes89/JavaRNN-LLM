package javallmsimple;

import java.io.Serializable;

/**
 *
 * @author Slam
 */
public class LSTMState implements Serializable {

    private static final long serialVersionUID = 1L;
    final double[] h;
    final double[] c;

    LSTMState(double[] h, double[] c) {
        this.h = h;
        this.c = c;
    }
}
