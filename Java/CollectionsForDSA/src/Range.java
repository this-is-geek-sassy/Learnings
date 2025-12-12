
import java.util.Iterator;
import java.util.NoSuchElementException;



public record Range(int start, int end) implements Iterable<Integer> {

    @Override
    public Iterator<Integer> iterator() {
        return new Iterator<>() {
            
            private int index = start;

            @Override
            public boolean hasNext() {
                return index < end;
            }

            @Override
            public Integer next() {
                if (index > end) {
                    throw new NoSuchElementException("" + index);
                }
                int currentIndex = index;
                index++;
                return currentIndex;
            }
        };
    }
}