import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.Collection;

public class Factory {

    public static void main(String[] args) {
        List<String> stringList = List.of("one", "two", "three");
        Set<String> stringSet = Set.of("one", "two", "three");

        System.out.println(stringList.getClass());
        System.out.println(stringSet.getClass());

        Collection<String> strings = Arrays.asList("one", "two", "three");

        List<String> list = List.copyOf(strings);
        Set<String> set = Set.copyOf(strings);

        System.out.println(strings.getClass());
        System.out.println(list.getClass());
        System.out.println(set.getClass());
        System.out.println(list);
        System.out.println(set);
    }
}
