import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Maps {

    public static void main(String[] args) {
        Map<String, Integer> inventory = new HashMap<>();
        inventory.put("apples", 50);
        inventory.put("bananas", 30);
        inventory.put("oranges", 25);

        System.out.println("inventory status: ");
        inventory.forEach((item, count) -> System.out.println(item + ": " + count));

        // compute - always executes, can handle null values
        inventory.compute("apples", (item, count) -> count != null ? count+20 : 20);

        inventory.computeIfPresent("bananas", (item, count) -> count-5);

        inventory.computeIfAbsent("grapes", item -> 15);

        System.out.println("After compute operations:");
        inventory.forEach((item, count) -> System.out.println(item + " :: " + count));

        // more compute operations:
        inventory.computeIfPresent("nonexistent", (item, count) -> count-9999);
        inventory.computeIfAbsent("peers", (item) -> 245);

        System.out.println("Final inventory:");
        inventory.forEach((item, count) -> System.out.println(item + " :: " + count));


        // merge patterns
        List<String> strings = List.of("one", "two", "three", "four", "five", "six", "seven");
        Map<Integer, String> map = new HashMap<>();
        for (String word: strings) {
            int length = word.length();
            map.merge(length, word, 
                    (existingValue, newWord) -> existingValue + ", " + newWord);
        }

        map.forEach((key, value) -> System.out.println(key + " :: " + value));

    }
}
