//
// The C++ side of the C++/Python interface
//
// The goal of this is to allow for sending out incoming data from the
// autopilot to all the connected clients (probably only one) and then receive
// back commands to send to the autopilot.
//
// How the connections will work:
// - Open: When a client connects, a TCP connection is opened
// - Send: When new data is available, it is sent out to all open connections
// - Receive: When the clients send commands back, they are sent to the autopilot
// - Close: The connection is closed when either the client or server finishes
//
// Based on:
// http://think-async.com/Asio/asio-1.10.6/doc/asio/tutorial/tutdaytime3/src.html
// http://think-async.com/Asio/asio-1.10.6/src/examples/cpp11/chat/chat_server.cpp
//
#include <set>
#include <atomic>
#include <ctime>
#include <mutex>
#include <thread>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <functional>
#include <asio.hpp>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

using asio::ip::tcp;

// Get current date string, for debugging
std::string now()
{
    std::time_t now = std::time(0);
    std::string date = ctime(&now);
    // Remove line return at the end
    date = date.substr(0, date.size()-1);

    return date;
}

//
// Data structure to handle the data we receive from the autopilot
//
struct AutopilotData
{
    std::string date;
    double lat;
    double lon;
    double alt;
    // ...

    AutopilotData()
        : date(now()), lat(0), lon(0), alt(1000.0)
    {
    }

    AutopilotData(const std::string& s)
    {
        rapidjson::Document j;
        j.Parse(s.c_str());
        date = j["date"].GetString();
        lat = j["lat"].GetDouble();
        lon = j["lon"].GetDouble();
        alt = j["alt"].GetDouble();
    }

    AutopilotData(const std::string& date, const double lat, const double lon,
            const double alt)
        : date(date), lat(lat), lon(lon), alt(alt)
    {
    }

    // Output JSON string
    /*{
        {"type", "data"},
        {"date", date},
        {"lat", lat},
        {"lon", lon},
        {"alt", alt}
    }*/
    std::string str() const
    {
        rapidjson::StringBuffer s;
        rapidjson::Writer<rapidjson::StringBuffer> writer(s);
        writer.StartObject();
        writer.Key("type"); writer.String("data");
        writer.Key("date"); writer.String(date.c_str());
        writer.Key("lat"); writer.Double(lat);
        writer.Key("lon"); writer.Double(lon);
        writer.Key("alt"); writer.Double(alt);
        writer.EndObject();

        return s.GetString();
    }

    // Cast to string by outputting JSON string
    operator std::string() const
    {
        return str();
    }
};

//
// Data structure to handle the commands we want to send to the autopilot
//
struct AutopilotCommand
{
    // Orbit center
    std::string date;
    double lat;
    double lon;
    double alt;

    // Orbit radius
    double radius;

    AutopilotCommand()
        : date(now()), lat(0), lon(0), alt(1000.0), radius(10.0)
    {
    }

    AutopilotCommand(const std::string& s)
    {
        rapidjson::Document j;
        j.Parse(s.c_str());
        date = j["date"].GetString();
        lat = j["lat"].GetDouble();
        lon = j["lon"].GetDouble();
        alt = j["alt"].GetDouble();
        radius = j["radius"].GetDouble();
    }

    AutopilotCommand(const double lat, const double lon, const double alt,
            const double radius)
        : lat(lat), lon(lon), alt(alt), radius(radius)
    {
    }

    // Output JSON string
    /*{
        {"type", "command"},
        {"date", date},
        {"lat", lat},
        {"lon", lon},
        {"alt", alt}
        {"radius", radius}
    }*/
    std::string str() const
    {
        rapidjson::StringBuffer s;
        rapidjson::Writer<rapidjson::StringBuffer> writer(s);
        writer.StartObject();
        writer.Key("type"); writer.String("command");
        writer.Key("date"); writer.String(date.c_str());
        writer.Key("lat"); writer.Double(lat);
        writer.Key("lon"); writer.Double(lon);
        writer.Key("alt"); writer.Double(alt);
        writer.Key("radius"); writer.Double(radius);
        writer.EndObject();

        return s.GetString();
    }

    // Cast to string by outputting JSON string
    operator std::string() const
    {
        return str();
    }
};


//
// An individual connection from a client
//
// Session inherits from this and will provide a real push() function
//
class Connection
{
public:
    virtual ~Connection() {}
    virtual void push(const AutopilotData& d) = 0;
};

//
// Keep a list of all connections and allow pushing data out to all of
// them
//
typedef std::shared_ptr<Connection> connection_ptr;

class ConnectionManager
{
    enum { max_recent_data = 100 };
    enum { max_commands = 100 };
    std::set<connection_ptr> connections;
    std::mutex data_mutex;
    std::deque<AutopilotData> recent_data;
    std::mutex commands_mutex;
    std::deque<AutopilotCommand> commands;

public:
    ConnectionManager()
    {
    }

    // Add/remove connections from the list
    void addConnection(connection_ptr con)
    {
        connections.insert(con);

        std::lock_guard<std::mutex> guard(data_mutex);

        for (auto d : recent_data)
            con->push(d);
    }

    void removeConnection(connection_ptr con)
    {
        connections.erase(con);
    }

    // Send new data to all connections
    void push(const AutopilotData& d)
    {
        std::lock_guard<std::mutex> guard(data_mutex);

        // Save new data
        recent_data.push_back(d);

        // Remove old data
        while (recent_data.size() > max_recent_data)
            recent_data.pop_front();

        // Push this data out to all the connections
        for (auto con : connections)
            con->push(d);
    }

    // When a connection sends a command, save it
    void processCommand(const AutopilotCommand& c)
    {
        std::lock_guard<std::mutex> guard(commands_mutex);
        commands.push_back(c);

        while (commands.size() > max_commands)
            commands.pop_front();

        // TODO make this do a callback?
    }

    // If using polling, see if there's any new commands available
    bool pullCommandAvailable()
    {
        std::lock_guard<std::mutex> guard(commands_mutex);
        return !commands.empty();
    }

    // Get the oldest command that we haven't already processed
    AutopilotCommand pullCommand()
    {
        std::lock_guard<std::mutex> guard(commands_mutex);
        AutopilotCommand c = commands.front();
        commands.pop_front();
        return c;
    }
};

//
// For each connection, we want to send out data as we get it from the
// autopilot and then receive back commands from the clients.
//
class Session
    : public Connection,
      public std::enable_shared_from_this<Session>
{
    tcp::socket socket;
    ConnectionManager& manager;
    asio::streambuf streambuf;
    std::deque<AutopilotData> write_data;

public:
    Session(tcp::socket socket, ConnectionManager& manager)
        : socket(std::move(socket)), manager(manager)
    {
    }

    // Start this session
    void start()
    {
        // Add the new connection to our list of connections
        manager.addConnection(shared_from_this());

        // Start reading from this TCP connection
        do_read();
    }

    // Add the data to the queue to send and start sending
    void push(const AutopilotData& d)
    {
        // If the queue is empty, we're not processing the data
        bool write_in_progress = !write_data.empty();

        // Add the new data to the queue
        write_data.push_back(d);

        // If we weren't already processing the written data, start doing that
        // with this new data we've added
        if (!write_in_progress)
            do_write();
    }

private:
    // Read commands from the clients
    void do_read()
    {
        auto self(shared_from_this());

        // Read from connection until an \0
        //
        // See: http://think-async.com/Asio/asio-1.10.6/doc/asio/overview/core/line_based.html
        asio::async_read_until(socket, streambuf, '\0',
                [this, self](std::error_code ec, std::size_t /*length*/)
                {
                    if (!ec)
                    {
                        // Input stream from this ASIO stream buf
                        std::istream is(&streambuf);

                        // Get up to the \0
                        std::string command;
                        std::getline(is, command, '\0');

                        // Do something with the command we read
                        manager.processCommand(command);

                        // Wait for another message
                        do_read();
                    }
                    else
                    {
                        std::cerr << "Error: reading, closing connection" << std::endl;

                        // An error occured, so remove this current
                        // connection
                        manager.removeConnection(shared_from_this());
                    }
                }
        );
    }

    // Write autopilot data to the connection
    void do_write()
    {
        auto self(shared_from_this());
        std::string data = write_data.front().str() + '\0';
        asio::async_write(socket,
                asio::buffer(data, data.length()),
                [this, self](std::error_code ec, std::size_t /*length*/)
                {
                    if (!ec)
                    {
                        // Remove from the queue since now we've written
                        // this data
                        write_data.pop_front();

                        // If the queue isn't empty, then write the next
                        // data as well
                        if (!write_data.empty())
                            do_write();
                    }
                    else
                    {
                        std::cerr << "Error: writing, closing connection" << std::endl;

                        // An error occured, so remove this current
                        // connection
                        manager.removeConnection(shared_from_this());
                    }
                }
        );
    }
};

//
// Listen on a port for incomming client connections
//
class Server
{
    tcp::acceptor acceptor;
    tcp::socket socket;
    ConnectionManager& manager;

public:
    Server(asio::io_service& io_service, int port, ConnectionManager&
            manager)
        : acceptor(io_service, tcp::endpoint(tcp::v4(), port)),
          socket(io_service),
          manager(manager)
    {
        do_accept();
    }

private:
    void do_accept()
    {
        acceptor.async_accept(socket,
                [this](std::error_code ec)
                {
                    if (!ec)
                        std::make_shared<Session>(std::move(socket),
                                manager)->start();

                    do_accept();
                });
    }
};

//
// Creates a thread connecting to the autopilot. In this demo, it'll just
// create demo data every second and print out sent data and received commands
// to the terminal.
//
class AutopilotThread
{
    std::atomic_bool exit;
    ConnectionManager& manager;
    std::thread t;

public:
    AutopilotThread(ConnectionManager& manager)
        : exit(false),
          // Note: manager *must* be initialized before starting the thread
          manager(manager),
          t(&AutopilotThread::run, this)
    {
    }

    // On destruct, end this
    ~AutopilotThread()
    {
        terminate();
        join();
    }

    // Our thread
    void run()
    {
        while (!exit)
        {
            // Send new data
            AutopilotData d;
            manager.push(d);

            std::cout << "Sent: " << d.str() << std::endl;

            // Get new commands if available
            while (manager.pullCommandAvailable())
            {
                AutopilotCommand c = manager.pullCommand();
                std::cout << "Received: " << c.str() << std::endl;
            }

            // Wait a little bit before sending more data
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }

    // Exit when we get the chance
    void terminate()
    {
        exit = true;
    }

    // Wait for the thread to complete
    void join()
    {
        t.join();
    }
};

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 2)
        {
            std::cerr << "Usage: server <port>" << std::endl;
            return 1;
        }

        // Port we want to run the server on
        int port = std::stoi(argv[1]);

        // Manage data between autopilot and network clients
        ConnectionManager manager;

        // Run demo autopilot
        AutopilotThread autopilotThread(manager);

        // Run network server
        asio::io_service io_service;
        Server server(io_service, port, manager);

        // Wait for server tasks to complete
        io_service.run();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
