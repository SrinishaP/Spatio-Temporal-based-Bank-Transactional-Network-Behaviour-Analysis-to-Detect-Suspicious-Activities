-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: May 14, 2023 at 07:27 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `money_laundering`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `alert_info`
--

CREATE TABLE `alert_info` (
  `id` int(11) NOT NULL,
  `message` varchar(200) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `alert_info`
--

INSERT INTO `alert_info` (`id`, `message`, `dtime`) VALUES
(1, 'Amount Rs. 2068118.36, debited from C1672125863 to C559485820, Previous balance: 5122403.81, after debit:3054285.4499999993', '2023-05-14 12:15:57'),
(2, 'Amount Rs. 195351.76, credited from C1420261366 to C639808302, Previous balance: 0.0, after credit:195351.76', '2023-05-14 12:16:14'),
(3, 'Amount Rs. 31074.0, credited from C457962127 to C614104997, Previous balance: 0.0, after credit:31074.0', '2023-05-14 12:16:36'),
(4, 'Amount Rs. 448025.06, debited from C348303891 to C1432596631, Previous balance: 519704.99, after debit:71679.93', '2023-05-14 12:16:49'),
(5, 'Amount Rs. 1448630.38, credited from C101899927 to C679483948, Previous balance: 0.0, after credit:1448630.38', '2023-05-14 12:18:37'),
(6, 'Amount Rs. 1639676.27, credited from C15998296 to C1652158314, Previous balance: 0.0, after credit:1639676.27', '2023-05-14 12:18:48'),
(7, 'Amount Rs. 785323.0, credited from C1204079316 to C637091706, Previous balance: 0.0, after credit:785323.0', '2023-05-14 12:18:59'),
(8, 'Amount Rs. 489767.32, credited from C869780206 to C926230389, Previous balance: 0.0, after credit:489767.32', '2023-05-14 12:19:25'),
(9, 'Amount Rs. 195351.76, credited from C1420261366 to C639808302, Previous balance: 0.0, after credit:195351.76', '2023-05-14 12:20:18'),
(10, 'Amount Rs. 1639676.27, credited from C15998296 to C1652158314, Previous balance: 0.0, after credit:1639676.27', '2023-05-14 12:21:01'),
(11, 'Amount Rs. 963532.14, debited from C430329518 to C991505714, Previous balance: 1095914.71, after debit:132382.56999999995', '2023-05-14 12:21:43'),
(12, 'Amount Rs. 963532.14, debited from C430329518 to C991505714, Previous balance: 1095914.71, after debit:132382.56999999995', '2023-05-14 12:53:03');

-- --------------------------------------------------------

--
-- Table structure for table `reg_mgr`
--

CREATE TABLE `reg_mgr` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(30) NOT NULL,
  `branch` varchar(20) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `reg_mgr`
--

INSERT INTO `reg_mgr` (`id`, `name`, `mobile`, `email`, `branch`, `uname`, `pass`, `status`) VALUES
(1, 'Kumar', 9638527415, 'kumar@gmail.com', 'Chennai', 'kumar', '1234', 0);
