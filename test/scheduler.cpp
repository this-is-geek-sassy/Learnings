#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

struct Task
{
    int id;
    std::string description;
    bool completed;
    time_t created;

    Task(int id, const std::string &desc)
        : id(id), description(desc), completed(false), created(time(nullptr)) {}
};

class TaskScheduler
{
private:
    std::vector<Task> tasks;
    std::string filename;
    int nextId;

    void loadTasks()
    {
        std::ifstream file(filename);
        if (!file.is_open())
            return;

        std::string line;
        while (getline(file, line))
        {
            if (line.empty())
                continue;

            std::stringstream ss(line);
            int id, completed;
            time_t created;
            std::string desc;

            ss >> id >> completed >> created;
            getline(ss, desc);

            if (!desc.empty() && desc[0] == ' ')
            {
                desc = desc.substr(1);
            }

            Task task(id, desc);
            task.completed = completed;
            task.created = created;
            tasks.push_back(task);

            if (id >= nextId)
            {
                nextId = id + 1;
            }
        }
        file.close();
    }

    void saveTasks()
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not save tasks to file." << std::endl;
            return;
        }

        for (const auto &task : tasks)
        {
            file << task.id << " "
                 << task.completed << " "
                 << task.created << " "
                 << task.description << std::endl;
        }
        file.close();
    }

    std::string formatTime(time_t t)
    {
        char buffer[80];
        struct tm *timeinfo = localtime(&t);
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        return std::string(buffer);
    }

public:
    TaskScheduler(const std::string &file = ".tasks.dat")
        : filename(file), nextId(1)
    {
        loadTasks();
    }

    ~TaskScheduler()
    {
        saveTasks();
    }

    void addTask(const std::string &description)
    {
        if (description.empty())
        {
            std::cout << "Error: Task description cannot be empty." << std::endl;
            return;
        }

        Task task(nextId++, description);
        tasks.push_back(task);
        std::cout << "\033[32m✓\033[0m Task added: [" << task.id << "] " << description << std::endl;
        saveTasks();
    }

    void displayTasks()
    {
        if (tasks.empty())
        {
            std::cout << "\n\033[33mNo tasks scheduled.\033[0m" << std::endl;
            return;
        }

        std::cout << "\n╔═════╦══════╦═════════════════════╦═══════════════════════════════════════" << std::endl;
        std::cout << "║ ID  ║ Done ║ Created             ║ Description" << std::endl;
        std::cout << "╠═════╬══════╬═════════════════════╬═══════════════════════════════════════" << std::endl;

        for (const auto &task : tasks)
        {
            std::cout << "║ " << std::setw(3) << task.id << " ║ ";

            if (task.completed)
            {
                std::cout << "\033[32m ✓  \033[0m";
            }
            else
            {
                std::cout << "    ";
            }

            std::cout << " ║ " << formatTime(task.created) << " ║ ";

            if (task.completed)
            {
                std::cout << "\033[90m\033[9m" << task.description << "\033[0m";
            }
            else
            {
                std::cout << task.description;
            }
            std::cout << std::endl;
        }
        std::cout << "╚═════╩══════╩═════════════════════╩═══════════════════════════════════════" << std::endl;

        int completed = std::count_if(tasks.begin(), tasks.end(),
                                      [](const Task &t)
                                      { return t.completed; });
        std::cout << "\n\033[36mTotal:\033[0m " << tasks.size()
                  << " | \033[32mCompleted:\033[0m " << completed
                  << " | \033[33mPending:\033[0m " << (tasks.size() - completed) << std::endl;
    }

    void completeTask(int id)
    {
        auto it = std::find_if(tasks.begin(), tasks.end(),
                               [id](const Task &t)
                               { return t.id == id; });

        if (it == tasks.end())
        {
            std::cout << "\033[31mError:\033[0m Task with ID " << id << " not found." << std::endl;
            return;
        }

        if (it->completed)
        {
            std::cout << "\033[33mWarning:\033[0m Task " << id << " is already completed." << std::endl;
        }
        else
        {
            it->completed = true;
            std::cout << "\033[32m✓\033[0m Task " << id << " marked as complete." << std::endl;
            saveTasks();
        }
    }

    void removeTask(int id)
    {
        auto it = std::find_if(tasks.begin(), tasks.end(),
                               [id](const Task &t)
                               { return t.id == id; });

        if (it == tasks.end())
        {
            std::cout << "\033[31mError:\033[0m Task with ID " << id << " not found." << std::endl;
            return;
        }

        std::cout << "\033[32m✓\033[0m Removed: [" << it->id << "] " << it->description << std::endl;
        tasks.erase(it);
        saveTasks();
    }

    void clearCompleted()
    {
        int count = 0;
        auto it = tasks.begin();
        while (it != tasks.end())
        {
            if (it->completed)
            {
                it = tasks.erase(it);
                count++;
            }
            else
            {
                ++it;
            }
        }

        if (count > 0)
        {
            std::cout << "\033[32m✓\033[0m Cleared " << count << " completed task(s)." << std::endl;
            saveTasks();
        }
        else
        {
            std::cout << "\033[33mNo completed tasks to clear.\033[0m" << std::endl;
        }
    }
};

void clearScreen()
{
    std::system("clear");
}

void printHeader()
{
    std::cout << "\033[1;36m" << std::endl;
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           TASK SCHEDULER - Interactive Mode                   ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\033[0m" << std::endl;
}

void printHelp()
{
    std::cout << "\n\033[1;33mAvailable Commands:\033[0m" << std::endl;
    std::cout << "  \033[36madd\033[0m <description>    - Add a new task" << std::endl;
    std::cout << "  \033[36mlist\033[0m                 - Display all tasks" << std::endl;
    std::cout << "  \033[36mcomplete\033[0m <id>         - Mark task as complete" << std::endl;
    std::cout << "  \033[36mremove\033[0m <id>           - Remove a task" << std::endl;
    std::cout << "  \033[36mclear\033[0m                - Clear all completed tasks" << std::endl;
    std::cout << "  \033[36mcls\033[0m                  - Clear screen" << std::endl;
    std::cout << "  \033[36mhelp\033[0m                 - Show this help message" << std::endl;
    std::cout << "  \033[36mexit\033[0m / \033[36mquit\033[0m          - Exit the program" << std::endl;
}

void processCommand(TaskScheduler &scheduler, const std::string &input)
{
    std::stringstream ss(input);
    std::string command;
    ss >> command;

    if (command.empty())
    {
        return;
    }

    if (command == "add")
    {
        std::string description;
        getline(ss, description);
        if (!description.empty() && description[0] == ' ')
        {
            description = description.substr(1);
        }

        if (description.empty())
        {
            std::cout << "\033[31mError:\033[0m Task description required." << std::endl;
            std::cout << "Usage: add <description>" << std::endl;
        }
        else
        {
            scheduler.addTask(description);
        }
    }
    else if (command == "list")
    {
        scheduler.displayTasks();
    }
    else if (command == "complete")
    {
        int id;
        if (ss >> id)
        {
            scheduler.completeTask(id);
        }
        else
        {
            std::cout << "\033[31mError:\033[0m Task ID required." << std::endl;
            std::cout << "Usage: complete <id>" << std::endl;
        }
    }
    else if (command == "remove")
    {
        int id;
        if (ss >> id)
        {
            scheduler.removeTask(id);
        }
        else
        {
            std::cout << "\033[31mError:\033[0m Task ID required." << std::endl;
            std::cout << "Usage: remove <id>" << std::endl;
        }
    }
    else if (command == "clear")
    {
        scheduler.clearCompleted();
    }
    else if (command == "cls")
    {
        clearScreen();
        printHeader();
        scheduler.displayTasks();
    }
    else if (command == "help")
    {
        printHelp();
    }
    else if (command == "exit" || command == "quit")
    {
        std::cout << "\n\033[32mSaving tasks and exiting...\033[0m" << std::endl;
        exit(0);
    }
    else
    {
        std::cout << "\033[31mError:\033[0m Unknown command '" << command << "'" << std::endl;
        std::cout << "Type 'help' for available commands." << std::endl;
    }
}

int main()
{
    TaskScheduler scheduler;

    clearScreen();
    printHeader();
    scheduler.displayTasks();
    printHelp();

    std::string input;

    while (true)
    {
        std::cout << "\n\033[1;32m>\033[0m ";
        if (!getline(std::cin, input))
        {
            break;
        }

        processCommand(scheduler, input);
    }

    return 0;
}
