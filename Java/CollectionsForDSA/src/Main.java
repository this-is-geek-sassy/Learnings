import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.Predicate;
// import Range;

class User {
    private String name;
    public User() {

    }
    public User(String name) {
        this.name = name;
    }
}

public class Main {
    public static void main(String[] args) {

//        Collection<String> strings = new ArrayList<>();
//        strings.add("Hello");
//        strings.add("World");
//        System.out.println("strings = " + strings);
//        strings.remove("Hello");
//        System.out.println("strings = " + strings);
//
//        User user = new User("Rebecca");
//        System.out.println(strings.contains("World"));
//        System.out.println(strings.contains(user));
//
//        strings.remove("World");
//
//        strings.add("one");
//        strings.add("two");
//        strings.add("three");
//        strings.add("four");
//
//        Collection<String> first = new ArrayList<>();
//        first.add("one");
//        first.add("two");
//
//        Collection<String> second = new ArrayList<>();
//        second.add("one");
//        second.add("four");
//
//        System.out.println("Is first contained in strings? " + strings.containsAll(first));
//        System.out.println("Is second contained in strings? " + strings.containsAll(second));
//
//        String[] allStrings = strings.toArray(new String[] {});
//        String[] extraStrings = strings.toArray(new String[15]);
//
//        for (String elem : allStrings) {
//            System.out.printf("%s\t", elem);
//        }
//        System.out.println();
//
//        for (String elem : extraStrings) {
//            System.out.printf("%s\t", elem);
//        }
//        System.out.println();

        Predicate<String> isNull = Objects::isNull;
        Predicate<String> isEmpty = String::isEmpty;
        Predicate<String> isNullOrEmpty = isNull.or(isEmpty);
        Predicate<String> always = o -> true;

        Collection<String> strings = new ArrayList<>();
        strings.add(null);
        strings.add("");
        strings.add("one");
        strings.add("two");
        strings.add("");
        strings.add("three");
        strings.add(null);

        System.out.println("Strings: " + strings);
        strings.removeIf(isNullOrEmpty);
        System.out.println("Strings: " + strings);

        strings.add("four");
        strings.add("five");
        strings.add("six");
        strings.add("seven");


        for (Iterator<String> it = strings.iterator(); it.hasNext();) {
            
            String element = it.next();
            // strings.remove(element);
            strings.removeIf(isEmpty);
        }
        // System.out.println(strings);

        // for (int i : new Range(0, 5)) {
        //     System.out.println(i);
        // }
    }
}